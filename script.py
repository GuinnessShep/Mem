from math import floor
import time

from modules import shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length
from annoy import AnnoyIndex
import spacy
from collections import deque

from extensions.annoy_ltm.helpers import *
from extensions.annoy_ltm.metadata import check_hashes, compute_hashes, load_metadata, save_metadata
from extensions.annoy_ltm.embeddings import generate_embeddings
from extensions.annoy_ltm.keyword_tally import KeywordTally
from extensions.annoy_ltm.turn_templates import get_turn_templates, apply_turn_templates_to_rows

# parameters which can be customized in settings.json of webui
params = {
    'annoy_output_dir': "extensions/annoy_ltm/outputs/",
    'logger_level': 1, # higher number is more verbose logging. 3 is really as high as any reasonable person should go for normal debugging
    'vector_dim_override': -1, # magic number determined by your loaded model. This parameter is here so that should some style of model in the future not include the hidden_size in the config, this can be used as a workaround.
    'memory_retention_threshold': 0.68, # 0-1, lower value will make memories retain longer but can cause stack to overflow and irrelevant memories to be held onto
    'full_memory_additional_weight': 0.5, # 0-1, smaller value is more weight here.
    'num_memories_to_retrieve': 5, # the number of related memories to retrieve for the full message and every keyword group generated from the message. Can cause significant slowdowns.
    'keyword_grouping': 4, # the number to group keywords into. Higher means harder to find an exact match, which makes matches more useful to context but too high and no memories will be returned.
    'keyword_rarity_weight': 1, # Throttles the weight applied to memories favoring unique phrases and vocabulary.
    'maximum_memory_stack_size': 50, # just a cap on the stack so it doesn't blow.
    'prompt_memory_ratio': 0.4 # the ratio of prompt after the character context is applied that will be dedicated for memories.
}

#--------------- Logger ---------------
def logger(msg: str, lvl=5):
    if params['logger_level'] >= lvl:
        print(msg)

#--------------- Spacy NLP ---------------
nlp = spacy.load("en_core_web_sm", disable=["parser"])

#--------------- Custom Prompt Generator ---------------

class ChatGenerator:
    def __init__(self):
        self.memory_stack = deque()
        self.keyword_tally = KeywordTally()

    #--------------- Hidden Size Helper -------------
    def _get_hidden_size(self):
        if params['vector_dim_override'] != -1:
            return params['vector_dim_override']
        
        try:
            return shared.model.model.config.hidden_size
        except AttributeError:
            return len(generate_embeddings('generate a set of embeddings to determine the size of the result list', logger=logger))


    def preprocess_and_extract_keywords(self, text):
        text_to_process = remove_username_and_timestamp(text)
        # Tokenization, lowercasing, and stopword removal
        tokens = [token.text.lower() for token in nlp(text_to_process) if not token.is_stop]

        # Lemmatization
        lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]

        # Named Entity Recognition
        doc = nlp(text_to_process)
        named_entities = [ent.text for ent in doc.ents]

        keywords = lemmatized_tokens + named_entities

        return keywords
        
    #--------------- Memory ---------------
    def evaluate_memory_relevance(self, memory, conversation, min_relevance_threshold=0.2):
        memory_text = ''.join([user_mem + '\n' + bot_mem for user_mem, bot_mem in memory])
        conversation_text = ''.join(conversation)
        logger(f"evaluating memory relevance for memory: {memory}", 4)
        memory_keywords = " ".join(filter_keywords(self.preprocess_and_extract_keywords(memory_text)))
        conversation_keywords = " ".join(filter_keywords(self.preprocess_and_extract_keywords(conversation_text)))
        logger(f"comparing keywords {memory_keywords}\nagainst conversation {conversation_keywords}", 5)
        memory_embeddings = generate_embeddings(memory_keywords, logger=logger)
        conversation_embeddings = generate_embeddings(conversation_keywords, logger=logger)
        logger(f"memory_embeddings: {memory_embeddings}", 6)
        logger(f"conversation_embeddings: {conversation_embeddings}", 6)
        logger(f"len memory_embeddings: {len(memory_embeddings)}", 6)
        logger(f"len conversation_embeddings: {len(conversation_embeddings)}", 6)
        cosine_similarity_value = cosine_similarity(memory_embeddings, conversation_embeddings)
        logger(f"manually computed cosine similarity: {cosine_similarity_value}", 5)
        return cosine_similarity_value >= min_relevance_threshold


    def retrieve_related_memories(self, annoy_index, input_messages, history_rows, index_to_history_position, keyword_tally, num_related_memories=3, weight=0.5):
        return_memories = set()
        for input_str in input_messages:
            logger(f"retrieving memories for <input> {input_str} </input>", 3)
            if num_related_memories == 0:
                num_related_memories = annoy_index.get_n_items()
            input_embedding = generate_embeddings(remove_username_and_timestamp(input_str), logger=logger)
            results_indices = []
            results_distances = []

            # Query for the original input_embedding
            indices, distances = annoy_index.get_nns_by_vector(input_embedding, num_related_memories, include_distances=True)
            results_indices.extend(indices)
            results_distances.extend(distances)
            original_input_results_count = len(results_distances)

            # Get keywords
            keywords = self.preprocess_and_extract_keywords(input_str)
            filtered_keywords = filter_keywords(keywords)
            keyword_groups = generate_keyword_groups(filtered_keywords, params['keyword_grouping'])
            logger(f"INPUT_KEYWORDS: {','.join(filtered_keywords)}", 4)

            # Query for each keyword_embedding
            for keyword in keyword_groups:
                keyword_embedding = generate_embeddings(keyword, logger=logger)
                logger(f"looking up keyword \"{keyword}\" embeddings {keyword_embedding}", 5)
                indices, distances = annoy_index.get_nns_by_vector(keyword_embedding, num_related_memories, include_distances=True)
                logger(f"keyword matches: {keyword}\n{indices}\n{distances}", 5)
                results_indices.extend(indices)
                results_distances.extend(distances)

            if len(results_indices) == 0:
                return [] # If we don't have any results, not much point in progressing.

            # 1. Combine the results
            indices_distances = list(zip(results_indices, results_distances))

            # 2. Apply the weight to the original input distances
            for i in range(original_input_results_count):
                indices_distances[i] = (indices_distances[i][0], indices_distances[i][1] * weight)

            # 3. Create a new list of unique history positions tupled with their distance while applying weights for duplicates
            history_positions_distances = {}
            for index, distance in indices_distances:
                history_position = index_to_history_position[index]
                if history_position in history_positions_distances:
                    history_positions_distances[history_position].append(distance)
                else:
                    history_positions_distances[history_position] = [distance]

            
            weighted_history_positions = [(pos, min(distances) / len(distances)) for pos, distances in history_positions_distances.items()]

            return_memories.update(set(weighted_history_positions))
            # return_memories.extend(weighted_history_positions)

        # 4. Get the related memories using the new sorted list
        related_memories = [(pos, shared.history['internal'][max(0, pos - 1):pos + 1], distance) for pos, distance in list(return_memories)]

        # Get keywords for each memory and calculate their significance
        for i in range(len(related_memories)):
            index, memory, distance = related_memories[i]
            memory_keywords = []
            for user_msg, bot_reply in memory:
                memory_keywords.extend(filter_keywords(self.preprocess_and_extract_keywords(user_msg)))
                memory_keywords.extend(filter_keywords(self.preprocess_and_extract_keywords(bot_reply)))

            significance = params['keyword_rarity_weight'] * keyword_tally.get_significance(memory_keywords)
            logger(f"keywords [{','.join(memory_keywords)}] significance calculated at {significance}", 4)

            # Apply the significance ratio to the memory's distance value
            related_memories[i] = (index, memory, distance * significance)

        # 5. Sort the new list
        sorted_weighted_related_memories = sorted(related_memories, key=lambda x: (x[2], x[0]))
        logger(f"RESULTS: {sorted_weighted_related_memories}", 4)

        # 6. Filter out memories that are already present in the history added to the prompt
        non_duplicate_memories = [
            (index, memory, distance) for index, memory, distance in sorted_weighted_related_memories
            if all(msg not in history_rows for msg in memory)
        ]

        return non_duplicate_memories



    def build_memory_rows(self, history_rows, user_input, max_memory_length, turn_templates, relevance_threshold=0.2):
        user_turn, bot_turn = turn_templates

        # Filter out irrelevant memories
        logger(f"HISTORY_ROWS:{history_rows}", 5)
        conversation = [remove_username_and_timestamp(row) for row in history_rows] + [remove_timestamp(user_input)]
        logger(f"CONVERSATION:{conversation}", 5)

        def log_and_check_relevance(memory_tuple, conversation, relevance_threshold):
            relevance_check = self.evaluate_memory_relevance(memory_tuple[1], conversation, relevance_threshold)
            logger(f"\nrelevance_check: {relevance_check}\nmemory_tuple: {memory_tuple}", 4)
            return relevance_check

        # Use the log_and_check_relevance function in the list comprehension
        new_memory_stack = [memory_tuple for memory_tuple in self.memory_stack if log_and_check_relevance(memory_tuple, conversation, relevance_threshold)]
        new_memory_stack = new_memory_stack[params['maximum_memory_stack_size']:]

        logger(f"MEMORY_STACK:{new_memory_stack}", 5)
        logger(f"MEMORY_STACK SIZE: {len(new_memory_stack)}", 3)

        # Create memory_rows

        memory_len = 0
        memory_index = 0
        returned_memories = 0
        memory_rows = []
        last_index = 0
        last_memory_rows_count = 0

        while memory_index < len(new_memory_stack) and memory_len < max_memory_length:
            index, memory, _ = new_memory_stack[memory_index]
            new_memory_len = memory_len

            i = len(memory) - 1
            while i >= 0 and new_memory_len < max_memory_length:
                row = memory[i]
                new_memory_len += len(row) + 1
                i -= 1

            if new_memory_len > max_memory_length:
                if returned_memories == 0:
                    logger(f"memory length {memory_len} max memory length {max_memory_length}", 2)
                    # We have room for something so we should return something
                    last_index = i + 1
                    memory_rows.extend(memory[last_index:])
                    returned_memories = last_memory_rows_count - len(memory[last_index:])
            else:
                memory_len = new_memory_len
                last_memory_rows_count = len(memory)
            memory_index += 1

        if memory_index < len(new_memory_stack):
            logger(f"Hit max memory length of {max_memory_length} before adding all memories", 2)
            logger(f"Memory index: {memory_index} Memory length: {memory_len}", 2)
        elif memory_len >= max_memory_length:
            logger(f"Max memory length of {max_memory_length} reached", 2)
            logger(f"Memory index: {memory_index} Memory length: {memory_len}", 2)

        if memory_len < max_memory_length:
            memory_rows.extend([user_turn, bot_turn] + [row for _, row, _ in new_memory_stack[memory_index:]])

        self.memory_stack = deque(new_memory_stack[memory_index:])
        logger(f"MEMORY_ROWS:{memory_rows}", 5)
        return memory_rows


    #--------------- Generate ---------------
    def generate(self, user_input):
        user_input = remove_username_and_timestamp(user_input)

        # Process user input and extract keywords
        keywords = self.preprocess_and_extract_keywords(user_input)

        # Update the keyword tally
        self.keyword_tally.update(keywords)

        # Update the memory stack
        self.memory_stack.append((user_input, ""))

        # Generate the response
        response = shared.model.generate(
            user_input,
            memory=self.memory_stack,
            keyword_tally=self.keyword_tally,
            params=params
        )[0]

        # Extract the bot reply from the generated response
        bot_reply = response['generated_text']

        # Update the memory stack with the bot reply
        self.memory_stack[-1] = (user_input, bot_reply)

        return bot_reply


#--------------- Main ---------------
if __name__ == "__main__":
    shared.init()

    # Load the Annoy index and metadata
    annoy_index = AnnoyIndex(params['vector_dim_override'])
    annoy_index.load(f"{params['annoy_output_dir']}annoy.index")

    metadata = load_metadata()

    if not check_hashes(annoy_index, metadata):
        logger("Index and metadata hashes do not match. Regenerating index...")
        compute_hashes(annoy_index, metadata)
        annoy_index.unload()
        annoy_index.load(f"{params['annoy_output_dir']}annoy.index")
        save_metadata(metadata)
        logger("Index regenerated successfully.")

    # Create a new instance of the chat generator
    chat_generator = ChatGenerator()

    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break

        # Generate the bot's response
        bot_reply = chat_generator.generate(user_input)
        print("Bot:", bot_reply)
        print()
