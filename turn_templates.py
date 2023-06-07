from extensions.annoy_ltm.helpers import replace_all

def get_turn_templates(state, is_instruct, logger):

    logger(f"state['turn_template']: {state['turn_template']}", 5)
    
    # Building the turn templates
    if 'turn_template' not in state or state['turn_template'] == '':
        if is_instruct:
            template = '\n<|user-message|>\n\n<|bot-message|>\n'
        else:
            template = ': <|user-message|>\n: <|bot-message|>\n'
    else:
        template = state['turn_template'].replace(r'\n', '\n')

    replacements = {
        '<|name1|>': state['name1'].strip(),
        '<|name2|>': state['name2'].strip(),
    }
    logger(f"turn_template replacements: {replacements}", 5)

    user_turn = replace_all(template.split('<|user-message|>')[0], replacements)
    bot_turn = replace_all(template.split('<|bot-message|>')[1], replacements)
    user_turn_stripped = replace_all(user_turn.split(':')[1].strip(), replacements)
    bot_turn_stripped = replace_all(bot_turn.split(':')[1].strip(), replacements)

    logger(f"turn_templates:\nuser_turn:{user_turn}\nbot_turn:{bot_turn}\nuser_turn_stripped:{user_turn_stripped}\nbot_turn_stripped:{bot_turn_stripped}", 5)

    return user_turn, bot_turn, user_turn_stripped, bot_turn_stripped

def apply_turn_templates_to_rows(rows, state, logger):
    is_instruct = state['mode'] == 'instruct'
    user_turn, bot_turn, user_turn_stripped, bot_turn_stripped = get_turn_templates(state, is_instruct, logger=logger)
    output_rows = []
    for i, row in enumerate(rows):
        if row[0] not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            user_row = replace_all(user_turn, {'<|user-message|>': row[0].strip(), '<|row-number|>': str(i)})
        else:
            user_row = row[0]
        bot_row = replace_all(bot_turn, {'<|bot-message|>': row[1].strip()})
        output_rows.append((user_row, bot_row))

    return output_rows
