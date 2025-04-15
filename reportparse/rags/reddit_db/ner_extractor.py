def extract_company(ner_output):
    return [e['word'] for e in ner_output if e['entity_group'] == 'ORG']