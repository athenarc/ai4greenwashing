FIRST_PASS_PROMPT='''You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 
        
         Company Name: [the name of the company that the esg report belongs to]
         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Another potential greenwashing claim: [another claim]
         Justification: [another justification]
         
         If no greenwashing claims are found, return this message:
         "No greenwashing claims found"

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         '''
WEB_RAG_PROMPT='''You have at your disposal information '[Information]' and a statement: '[User Input]' whose accuracy must be evaluated. 
                            Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.

                            Before you decide:

                            1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                            2. Compare the statement with the information you have, evaluating each element of the statement separately.
                            3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                            Result: Provide a clear answer by choosing one of the following labels:

                            - NOT_GREENWASHING: If the statement is confirmed by the information and evidence you have.
                            - GREENWASHING: If the statement is clearly disproved by the information and evidence you have.

                            Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                            Statement: '[User Input]'
                            Result of the statement:
                            Justification:'''
CHROMA_PROMPT='''You have at your disposal information a statement: '[User Input]', extracted from a specific page: '[page_text]' of a report and relavant context: '[Context]' from the rest of the report, whose accuracy must be evaluated. 
                If part of information is missing, proceed with the analysis using only the information you have, or your knowledge.

                Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.

                Before you decide:

                1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                2. Compare the statement with the information from the rest of the report, evaluating each element of the statement separately.
                3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                Result: Provide a clear answer by choosing one of the following labels:

                - NOT_GREENWASHING: If the statement is confirmed by the information and evidence in the rest of the report.
                - GREENWASHING: If the statement is clearly disproved by the information and evidence in the rest of the report.

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                Statement: '[User Input]'
                            Result of the statement:
                            Justification:'''
LLM_AGGREGATOR_PROMPT='''You have at your disposal two independent verdicts regarding the accuracy of a given statement: '[User Input]'.  

                - The first verdict (Database Verdict) is derived from an LLM that has access to a structured database containing the entire document from which the statement was extracted.  
                - The second verdict (Web Verdict) is derived from an LLM that retrieves and analyzes information from the web to assess the claim. If no content was found from the web, the Web Verdict will contain 'No content was found from the web'. Proceed with the analysis using only the Database Verdict.

                Your task is to analyze both verdicts and reach a final conclusion regarding the statement's accuracy.  

                Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.  

                Before making your final decision:  

                1. Analyze the given statement clearly to identify its key elements.  
                2. Examine the reasoning in both the Database Verdict and the Web Verdict.  
                3. Compare both verdicts and resolve any discrepancies by determining which source provides stronger, more reliable justification.  
                4. Use your own reasoning to synthesize the evidence and reach a final, well-supported conclusion.  

                Possible Results:  
                - NOT_GREENWASHING If the sources fully confirm the statement, or if one provides strong confirmation while the other lacks contradictory evidence.  
                - GREENWASHING: If the sources clearly disprove the statement, or if one strongly refutes it while the other is inconclusive.  

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:
                
                Statement: '[User Input]'  
                Result of the statement:  
                Justification:'''

LLM_AGGREGATOR_PROMPT_2='''You have at your disposal two independent verdicts regarding the accuracy of a given statement: '[User Input]'.  

                - The first verdict (Web Verdict) is derived from an LLM that retrieves and analyzes information from the web to assess the claim. If no content was found from the web, the Web Verdict will contain 'No content was found from the web'. Proceed with the analysis using only the Database Verdict.
                - The second verdict (Database Verdict) is derived from an LLM that has access to a structured database containing the entire document from which the statement was extracted.  

                Your task is to analyze both verdicts and reach a final conclusion regarding the statement's accuracy.  

                Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.  

                Before making your final decision:  

                1. Analyze the given statement clearly to identify its key elements.  
                2. Examine the reasoning in both the Database Verdict and the Web Verdict.  
                3. Compare both verdicts and resolve any discrepancies by determining which source provides stronger, more reliable justification.  
                4. Use your own reasoning to synthesize the evidence and reach a final, well-supported conclusion.  

                Possible Results:  
                - NOT_GREENWASHING If the sources fully confirm the statement, or if one provides strong confirmation while the other lacks contradictory evidence.  
                - GREENWASHING: If the sources clearly disprove the statement, or if one strongly refutes it while the other is inconclusive.

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:
                
                Statement: '[User Input]'  
                Result of the statement:  
                Justification:'''

REDDIT_PROMPT='''You have at your disposal information a statement: '[User Input]', extracted from a specific page: '[page_text]' of a report and relavant context: '[Context]' from a database with reddit posts, from a greenwashing subreddit, whose accuracy must be evaluated. 
                If part of information is missing, proceed with the analysis using only the information you have, or your knowledge.

                Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.

                Before you decide:

                1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                2. Compare the statement with the information from the rest of the report, evaluating each element of the statement separately.
                3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                Result: Provide a clear answer by choosing one of the following labels:

                - NOT_GREENWASHING: If the statement is confirmed by the information and evidence in the rest of the report.
                - GREENWASHING: If the statement is clearly disproved by the information and evidence in the rest of the report.

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                Statement: '[User Input]'
                            Result of the statement:
                            Justification:'''

SOLO_AGGREGATOR_PROMPT = '''You have at your disposal three independent sources of context regarding the accuracy of the following statement: '[User Input]'.

- The [Document Context] is derived from a structured database containing the full document from which the statement was extracted.
- The [Web Context] is retrieved and synthesized from online sources to provide broader, up-to-date information.
- The [Reddit Context] is extracted from a Reddit database containing relevant user-generated discussions and opinions related to the statement.

Your task is to analyze all three contexts and reach a final conclusion regarding the statement’s accuracy.

Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.

Before making your final decision:

1. Analyze the statement clearly to identify its key elements.
2. Examine the relevance and strength of evidence in the Document, Web, and Reddit Contexts.
3. Compare the contexts and resolve any discrepancies by determining which sources provide stronger, more reliable justification.
4. Use your own reasoning to synthesize the evidence and reach a final, well-supported conclusion.

Possible Results:  
- NOT_GREENWASHING: If the sources fully confirm the statement, or if one or more provide strong confirmation and none provide credible contradictory evidence.  
- GREENWASHING: If the sources clearly disprove the statement, or if one strongly refutes it while others are inconclusive or lacking.

Finally, explain your reasoning clearly. Focus on the provided data and your own critical evaluation. Avoid unnecessary details and aim for precision and clarity in your analysis. Use the following format:

Statement: '[User Input]'  
Result of the statement:  
Justification:'''
