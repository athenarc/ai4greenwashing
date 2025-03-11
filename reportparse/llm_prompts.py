FIRST_PASS_PROMPT='''You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 

         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Another potential greenwashing claim: [another claim]
         Justification: [another justification]
         
         If no greenwashing claims are found, return this message:
         "No greenwashing claims found"

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         State the claim like a search query. The query should be brief, precise, and focus on the core topics or keywords mentioned in the text. Avoid unnecessary words or long phrases, and aim for a search-friendly format.'''
WEB_RAG_PROMPT='''You have at your disposal information '[Information]' and a statement: '[User Input]' whose accuracy must be evaluated. 
                            Use only the provided information in combination with your knowledge to decide whether the statement is TRUE, FALSE, PARTIALLY TRUE, or PARTIALLY FALSE.

                            Before you decide:

                            1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                            2. Compare the statement with the information you have, evaluating each element of the statement separately.
                            3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                            Result: Provide a clear answer by choosing one of the following labels:

                            - TRUE: If the statement is fully confirmed by the information and evidence you have.
                            - FALSE: If the statement is clearly disproved by the information and evidence you have.
                            - PARTIALLY TRUE: If the statement contains some correct elements, but not entirely accurate.
                            - PARTIALLY FALSE: If the statement contains some correct elements but also contains misleading or inaccurate information.

                            Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                            Statement: '[User Input]'
                            Result of the statement:
                            Justification:'''
CHROMA_PROMPT='''You have at your disposal information a statement: '[User Input]', extracted from a specific page: '[page_text]' of a report and relavant context: '[Context]' from the rest of the report, whose accuracy must be evaluated. 
                            Use only the provided information in combination with your knowledge to decide whether the statement is TRUE, FALSE, PARTIALLY TRUE, or PARTIALLY FALSE.

                Use only the provided information in combination with your knowledge to decide whether the statement is TRUE, FALSE, PARTIALLY TRUE, or PARTIALLY FALSE.

                Before you decide:

                1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                2. Compare the statement with the information from the rest of the report, evaluating each element of the statement separately.
                3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                Result: Provide a clear answer by choosing one of the following labels:

                - TRUE: If the statement is fully confirmed by the information and evidence in the rest of the report.
                - FALSE: If the statement is clearly disproved by the information and evidence in the rest of the report.
                - PARTIALLY TRUE: If the statement contains some correct elements but is not entirely accurate.
                - PARTIALLY FALSE: If the statement contains some correct elements but also contains misleading or inaccurate information.

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                Statement: '[User Input]'
                            Result of the statement:
                            Justification:'''
LLM_AGGREGATOR_PROMPT='''You have at your disposal two independent verdicts regarding the accuracy of a given statement: '[User Input]'.  

                - The first verdict (Database Verdict) is derived from an LLM that has access to a structured database containing the entire document from which the statement was extracted.  
                - The second verdict (Web Verdict) is derived from an LLM that retrieves and analyzes information from the web to assess the claim.  

                Your task is to analyze both verdicts and reach a final conclusion regarding the statement's accuracy.  

                Use only the provided information in combination with your knowledge to decide whether the statement is TRUE, FALSE, PARTIALLY TRUE, or PARTIALLY FALSE.  

                Before making your final decision:  

                1. Analyze the given statement clearly to identify its key elements.  
                2. Examine the reasoning in both the Database Verdict and the Web Verdict.  
                3. Compare both verdicts and resolve any discrepancies by determining which source provides stronger, more reliable justification.  
                4. Use your own reasoning to synthesize the evidence and reach a final, well-supported conclusion.  

                Possible Results:  
                - TRUE If both sources fully confirm the statement, or if one provides strong confirmation while the other lacks contradictory evidence.  
                - FALSE: If both sources clearly disprove the statement, or if one strongly refutes it while the other is inconclusive.  
                - PARTIALLY TRUE: If the statement contains correct elements but is incomplete or slightly misleading.  
                - PARTIALLY FALSE: If the statement has some correct elements but is also significantly inaccurate or misleading.  

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:
                
                Statement: '[User Input]'  
                Result of the statement:  
                Justification:'''