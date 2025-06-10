import ollama
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

ENTITY_TYPES = [
    "Organization",
    "Environmental Claim",
    "Regulation",
    "Product",
    "Impact Metric",
    "Green Certification",
    "Date",
    "Location",
    "ESG",
    "Greenwashing",
    "Environmental standards",
    "Social standards",
    "Governance standards",
    "Person",
    "Accusation",
    "Accreditation",
    "Certification",
    "Compliance",
    "Violation",
    "Sustainability Initiative",
    "Technology",
]

PREDICATES = [
    "founded by",
    "acquired",
    "certified with",
    "claims",
    "reports",
    "complies with",
    "violates",
    "based in",
    "targets",
    "announced",
    "related to",
    "affects",
    "regulated by",
    "founded in",
    "supports",
    "develops",
    "accuses",
    "audited by",
    "denies",
    "investigated for",
    "achieves",
]


class triplet_extractor:

    def __init__(self):
        self.entity_types = ENTITY_TYPES
        self.predicates = PREDICATES

    # convert llm output to json format
    def clean_json_string(self, raw_str):
        raw_str = raw_str.strip()
        if raw_str.startswith("```json"):
            raw_str = raw_str.replace("```json", "").replace("```", "").strip()
        return raw_str

    # chunk text if it's too big for the llm and to increase accuracy
    def chunk_text(self, text, max_words=5000):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        word_count = 0

        for sentence in sentences:
            words_in_sentence = len(sentence.split())
            if word_count + words_in_sentence > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                word_count = words_in_sentence
            else:
                current_chunk.append(sentence)
                word_count += words_in_sentence

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # convert json string to (subject, predicate, object) triplets
    def json_to_triplets(self, json_str):

        json_str = self.clean_json_string(json_str)
        data = json.loads(json_str)
        lines = data["entities_and_triples"]
        id2entity = {}
        named_triplets = []

        # Patterns
        entity_pattern = re.compile(r"^\[(\d+)\],\s*(.*?):(.*)$")
        triple_pattern_entity = re.compile(r"^\[(\d+)\]\s+(\w+)\s+\[(\d+)\]$")
        triple_pattern_literal = re.compile(r"^\[(\d+)\]\s+(\w+)\s+(.*)$")

        for line in lines:
            line = line.strip()

            # Entity line
            entity_match = entity_pattern.match(line)
            if entity_match:
                idx, ent_type, ent_value = entity_match.groups()
                id2entity[idx] = ent_value.strip()
                continue

            # Triple with both subject and object as entities
            triple_match = triple_pattern_entity.match(line)
            if triple_match:
                subj, rel, obj = triple_match.groups()
                subj_name = id2entity.get(subj, f"[{subj}]")
                obj_name = id2entity.get(obj, f"[{obj}]")
                named_triplets.append((subj_name, rel, obj_name))
                continue

            # Triple with subject as entity and object as literal
            triple_match = triple_pattern_literal.match(line)
            if triple_match:
                subj, rel, obj = triple_match.groups()
                subj_name = id2entity.get(subj, f"[{subj}]")
                named_triplets.append((subj_name, rel, obj.strip()))
                continue

        return named_triplets

    # Function to extract triplets from an article text using LLM (this is for news articles only)
    def article_triplet_extractor(self, text):
        text = str(text).replace("\n", " ")
        chunks = self.chunk_text(text)

        all_triplets = []

        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---\n")
            prompt = f"""
            Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text.

            **Entity Types:**
            {', '.join(self.entity_types)}

            **Predicates:**
            {', '.join(self.predicates)}

            **Text:**
            {chunk}
            """

            response = ollama.chat(
                model="sciphi/triplex:latest",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )

            triplet_str = response["message"]["content"]

            try:
                triplets = self.json_to_triplets(triplet_str)
                all_triplets.extend(triplets)
            except Exception as e:
                print(f"Warning: Failed to parse chunk {i+1} - {e}")

        return all_triplets

    # function to extract only entities from the json format
    def extract_entities(self, json_data_string):
        json_data_string = self.clean_json_string(json_data_string)
        data = json.loads(json_data_string)
        entities_and_triples = data.get("entities_and_triples", [])
        entity_names = []
        for item in entities_and_triples:
            if ":" in item and "," in item.split("]")[1]:
                colon_index = item.find(":")
                if colon_index != -1:
                    entity_name = item[colon_index + 1 :].strip()
                    entity_names.append(entity_name)
        return entity_names

    # function for generic calling of the triplet extractor llm
    def call_triplet_llm(self, text):
        text = str(text).replace("\n", " ")
        chunks = self.chunk_text(text)

        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---\n")
            prompt = f"""
            Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text.

            **Entity Types:**
            {', '.join(self.entity_types)}

            **Predicates:**
            {', '.join(self.predicates)}

            **Text:**
            {chunk}
            """

            response = ollama.chat(
                model="sciphi/triplex:latest",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )

            triplet_str = response["message"]["content"]
            return triplet_str
