FIRST_PASS_PROMPT = """
You are a fact-checker who specializes in detecting greenwashing in corporate ESG statements and marketing claims.

Task:
Analyze the given text and identify whether it contains any greenwashing claims.

Greenwashing Definition:
A claim is considered greenwashing if at least one of the following is true:

Unreliable or misleading sustainability labels are used without clear substantiation.
(Example: H&M and Decathlon were prompted by the Netherlands Authority for Consumers and Markets to stop using labels such as "eco design" and "conscious" in 2022, because they did not clearly describe the criteria behind the labels.)

Legal requirements are presented as if they are voluntary sustainability efforts.
(Example: McDonald’s promoted its use of reusable cutlery as an environmental initiative, while it was in fact a legal obligation in France since January 2023.)

A green claim is made about the entire product, while in reality it applies only to a specific part or feature.
(Example: Kohl’s and Walmart advertised their products as "bamboo-based", implying sustainability, but the products were actually made from rayon — a material derived through a highly polluting process that eliminates bamboo’s environmental benefits. The FTC fined both companies for this misrepresentation.)

Environmental claims are made without supporting evidence or cannot be substantiated.
(Example: Keurig was fined by Canada’s Competition Bureau for claiming that its single-use coffee pods were recyclable, when in practice, recycling was not available in most municipalities and the claims could not be backed by data.)


Instructions:
Review the given ESG or marketing text and identify all potential greenwashing claims that meet one or more of the above criteria.

Return your findings using strictly the following format:
        
Company Name: [the name of the company that the ESG report or ad belongs to]  
Potential greenwashing claim: [text snippet of the greenwashing claim]  
Justification: [short explanation based on the criteria above]

Another potential greenwashing claim: [text snippet]  
Justification: [explanation]

... [repeat if applicable]
         
If no greenwashing claims are found, return only the following message:
"No greenwashing claims found"

Important:
Do not provide any commentary, reasoning outside of the provided criteria, or response formatting that deviates from the structure above.
"""

WEB_RAG_PROMPT = """
Task:
You are tasked with evaluating whether the following statement constitutes greenwashing. You have access to supporting evidence under the section labeled [Information], and the statement under [User Input].

Use only the provided information, combined with your general knowledge, to make your judgment. Do not speculate or use unverifiable facts.

Greenwashing Definition:
A claim is considered greenwashing if at least one of the following is true:

Unreliable or misleading sustainability labels are used without clear substantiation.
(Example: H&M and Decathlon were prompted by the Netherlands Authority for Consumers and Markets to stop using labels such as "eco design" and "conscious" in 2022, because they did not clearly describe the criteria behind the labels.)

Legal requirements are presented as if they are voluntary sustainability efforts.
(Example: McDonald’s promoted its use of reusable cutlery as an environmental initiative, while it was in fact a legal obligation in France since January 2023.)

A green claim is made about the entire product, while in reality it applies only to a specific part or feature.
(Example: Kohl’s and Walmart advertised their products as "bamboo-based", implying sustainability, but the products were actually made from rayon — a material derived through a highly polluting process that eliminates bamboo’s environmental benefits. The FTC fined both companies for this misrepresentation.)

Environmental claims are made without supporting evidence or cannot be substantiated.
(Example: Keurig was fined by Canada’s Competition Bureau for claiming that its single-use coffee pods were recyclable, when in practice, recycling was not available in most municipalities and the claims could not be backed by data.)

Before making a decision:

1. Break down the statement to identify its key claims.
2. Cross-check each part of the statement against the provided information.
3. Use your own knowledge only in support of what is confirmed by the data. Do not speculate or use unverifiable facts.

Label the statement using one of the following:

- GREENWASHING: if the statement is contradicted or undermined by the information provided.
- NOT_GREENWASHING: if the statement is supported or aligned with the information provided.


Respond in this format:

Statement: '[User Input]'  
Result of the statement: [GREENWASHING or NOT_GREENWASHING]  
Justification: [Brief and precise reasoning based on the evidence and knowledge]
"""

CHROMA_PROMPT = """
You are evaluating whether the following statement constitutes greenwashing. You have access to:

- The statement: '[User Input]'  
- The page it was extracted from: '[page_text]'  
- Additional relevant context from the report: '[Context]'

Use only the provided information, combined with your general knowledge, to make your judgment. Do not speculate or use unverifiable facts.

Greenwashing Definition:
A claim is considered greenwashing if at least one of the following is true:

Unreliable or misleading sustainability labels are used without clear substantiation.
(Example: H&M and Decathlon were prompted by the Netherlands Authority for Consumers and Markets to stop using labels such as "eco design" and "conscious" in 2022, because they did not clearly describe the criteria behind the labels.)

Legal requirements are presented as if they are voluntary sustainability efforts.
(Example: McDonald’s promoted its use of reusable cutlery as an environmental initiative, while it was in fact a legal obligation in France since January 2023.)

A green claim is made about the entire product, while in reality it applies only to a specific part or feature.
(Example: Kohl’s and Walmart advertised their products as "bamboo-based", implying sustainability, but the products were actually made from rayon — a material derived through a highly polluting process that eliminates bamboo’s environmental benefits. The FTC fined both companies for this misrepresentation.)

Environmental claims are made without supporting evidence or cannot be substantiated.
(Example: Keurig was fined by Canada’s Competition Bureau for claiming that its single-use coffee pods were recyclable, when in practice, recycling was not available in most municipalities and the claims could not be backed by data.)

Before making a decision:

1. Break down the statement to identify its key claims.
2. Cross-check each part of the statement against the provided information.
3. Use your own knowledge only in support of what is confirmed by the data. Do not speculate or use unverifiable facts.

Label your result as one of the following:

- GREENWASHING: if the statement is contradicted or undermined by the evidence in the report.
- NOT_GREENWASHING: if the statement is supported or consistent with the information provided.

Respond strictly in the following format:

Statement: '[User Input]'  
Result of the statement: [GREENWASHING or NOT_GREENWASHING]  
Justification: [Concise explanation based on the evidence and contextual information]
"""

LLM_AGGREGATOR_PROMPT = """
You are evaluating whether the following statement constitutes greenwashing: '[User Input]'

You have access to three independent verdicts:

- **Database Verdict**: Based on an LLM analysis of a structured database containing the full source document.
- **Web Verdict**: Based on an LLM analysis of external web search results. If no relevant content was found on the web, the Web Verdict will state: "No content was found from the web."
- **Reddit Context**: Based on an LLM analysis of relevant Reddit posts, focused on greenwashing-related content.

Use only the provided verdicts and your own knowledge to determine whether the statement is GREENWASHING or NOT_GREENWASHING.

Greenwashing Definition:
A claim is considered greenwashing if at least one of the following is true:

Unreliable or misleading sustainability labels are used without clear substantiation.
(Example: H&M and Decathlon were prompted by the Netherlands Authority for Consumers and Markets to stop using labels such as "eco design" and "conscious" in 2022, because they did not clearly describe the criteria behind the labels.)

Legal requirements are presented as if they are voluntary sustainability efforts.
(Example: McDonald’s promoted its use of reusable cutlery as an environmental initiative, while it was in fact a legal obligation in France since January 2023.)

A green claim is made about the entire product, while in reality it applies only to a specific part or feature.
(Example: Kohl’s and Walmart advertised their products as "bamboo-based", implying sustainability, but the products were actually made from rayon — a material derived through a highly polluting process that eliminates bamboo’s environmental benefits. The FTC fined both companies for this misrepresentation.)

Environmental claims are made without supporting evidence or cannot be substantiated.
(Example: Keurig was fined by Canada’s Competition Bureau for claiming that its single-use coffee pods were recyclable, when in practice, recycling was not available in most municipalities and the claims could not be backed by data.)

Important: If either the database or the web yields no relevant information, **ignore the absence** and proceed using the information that is available. Do not interpret a lack of evidence as support for or against the statement.

Before reaching your conclusion:

1. Break down the statement to identify its key claims.
2. Examine the reasoning in both the Database and Web Verdicts.
3. If they disagree, assess which provides stronger, clearer justification.
4. Use your own reasoning to synthesize the evidence and arrive at a well-supported conclusion.

Possible results:

- NOT_GREENWASHING: If the verdicts support the statement, or one provides strong support while the other is neutral or inconclusive.
- GREENWASHING: If the verdicts contradict the statement, or one provides strong refutation while the other is neutral or inconclusive.

Respond in the following format:

Statement: '[User Input]'  
Result of the statement: [GREENWASHING or NOT_GREENWASHING]  
Justification: [Concise explanation based on the verdicts and your reasoning]
"""

REDDIT_PROMPT = """
You are evaluating whether the following statement constitutes greenwashing: '[User Input]'

You have access to:

- The page the statement was extracted from: '[page_text]'
- Additional context: '[Context]', retrieved from a database of Reddit posts sourced from a greenwashing-focused subreddit

Use only the provided information, combined with your general knowledge, to make your judgment. Do not speculate or use unverifiable facts.

Greenwashing Definition:
A claim is considered greenwashing if at least one of the following is true:

Unreliable or misleading sustainability labels are used without clear substantiation.
(Example: H&M and Decathlon were prompted by the Netherlands Authority for Consumers and Markets to stop using labels such as "eco design" and "conscious" in 2022, because they did not clearly describe the criteria behind the labels.)

Legal requirements are presented as if they are voluntary sustainability efforts.
(Example: McDonald’s promoted its use of reusable cutlery as an environmental initiative, while it was in fact a legal obligation in France since January 2023.)

A green claim is made about the entire product, while in reality it applies only to a specific part or feature.
(Example: Kohl’s and Walmart advertised their products as "bamboo-based", implying sustainability, but the products were actually made from rayon — a material derived through a highly polluting process that eliminates bamboo’s environmental benefits. The FTC fined both companies for this misrepresentation.)

Environmental claims are made without supporting evidence or cannot be substantiated.
(Example: Keurig was fined by Canada’s Competition Bureau for claiming that its single-use coffee pods were recyclable, when in practice, recycling was not available in most municipalities and the claims could not be backed by data.)

Important: If some information is missing from the context or the Reddit database, proceed using what is available. Do **not** treat missing information as support for or against the statement.

Before making a decision:

1. Break down the statement to identify its key claims.
2. Cross-check the statement with the report content and Reddit-derived context.
3. Use your own reasoning and general knowledge **only** to support conclusions that are grounded in the provided information. Do not speculate or rely on unverifiable claims.

Label your conclusion using one of the following:

- NOT_GREENWASHING : If the statement is supported or confirmed by the provided context.
- GREENWASHING: If the statement is contradicted or undermined by the provided context.

Respond strictly in the following format:

Statement: '[User Input]'  
Result of the statement: [GREENWASHING or NOT_GREENWASHING]  
Justification: [Concise explanation based on the provided evidence and your reasoning]

"""

SOLO_AGGREGATOR_PROMPT = """
You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 

You are tasked with evaluating the accuracy of the following statement: '[User Input]'

You have access to three sources of supporting context:

- **Document Context**: Extracted from a structured database containing the full ESG report from which the statement originated.  
- **Web Context**: Retrieved and synthesized from online sources to provide broader, external information.  
- **Reddit Context**: Sourced from relevant Reddit posts, focused on greenwashing-related content.

Greenwashing Definition:
A claim is considered greenwashing if at least one of the following is true:

Unreliable or misleading sustainability labels are used without clear substantiation.
(Example: H&M and Decathlon were prompted by the Netherlands Authority for Consumers and Markets to stop using labels such as "eco design" and "conscious" in 2022, because they did not clearly describe the criteria behind the labels.)

Legal requirements are presented as if they are voluntary sustainability efforts.
(Example: McDonald’s promoted its use of reusable cutlery as an environmental initiative, while it was in fact a legal obligation in France since January 2023.)

A green claim is made about the entire product, while in reality it applies only to a specific part or feature.
(Example: Kohl’s and Walmart advertised their products as "bamboo-based", implying sustainability, but the products were actually made from rayon — a material derived through a highly polluting process that eliminates bamboo’s environmental benefits. The FTC fined both companies for this misrepresentation.)

Environmental claims are made without supporting evidence or cannot be substantiated.
(Example: Keurig was fined by Canada’s Competition Bureau for claiming that its single-use coffee pods were recyclable, when in practice, recycling was not available in most municipalities and the claims could not be backed by data.)

Use only the information provided—along with your general knowledge—to determine whether the statement is GREENWASHING or NOT_GREENWASHING.

Important: If any context is missing or lacks relevant information, proceed using what is available. Do **not** treat the absence of evidence as either supporting or refuting the statement.

Before making your final decision:

1. Break down the statement to identify its key claims.
2. Assess the relevance and strength of evidence in the Document, Web, and Reddit Contexts.
3. If sources conflict, determine which provides stronger, clearer justification.
4. Use your reasoning to synthesize the evidence and draw a final, well-supported conclusion.

Possible results:

- NOT_GREENWASHING: If the statement is supported by one or more sources and none provide credible contradictory evidence.
- GREENWASHING: If the statement is clearly contradicted by one or more sources, while others are inconclusive or lacking.

Respond strictly in the following format:

Statement: '[User Input]'  
Result of the statement: [GREENWASHING or NOT_GREENWASHING]  
Justification: [Concise explanation based on the provided evidence and your own reasoning]
"""
