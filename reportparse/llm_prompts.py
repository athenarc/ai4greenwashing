FIRST_PASS_PROMPT = """You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
        
         Greenwashing is defined as one of the following types
         
         - GREENWASHING TYPE 1:  using unreliable sustainability labels (e.g., H&M and Decathlon have used labels such as eco design and conscious
                                 and were prompted by the Netherlands Authority for Consumers and Markets to stop the practice in 2022 as the
                                 etailers did not clearly describe why the products received those labels).

         - GREENWASHING TYPE 2:  presenting legal requirements for the product as its distinctive features (e.g., McDonald’s advertises its reduction
                                 of plastic waste by using reusable cutlery, while this is a legal obligation since January 2023 in France)

         - GREENWASHING TYPE 3:  making green claims about the entire product when the claim applies only to a part/aspect of the product (e.g.,
                                 Kohl’s and Walmart were fined by the U.S. Federal Trade Commission for the misleading claim that their products
                                 were made of bamboo, a sustainable material, when they were made out of rayon, which is derived from bamboo
                                 via a highly toxic process that removes any environmental benefits of bamboo)

         - GREENWASHING TYPE 4:  making environmental claims for which the company cannot provide evidence (e.g., Keurig was fined by Canada’s
                                 Competition Bureau for misleading claims about the recycling process of its single-use coffee pods).      
                
         Your answer should follow the following format: 
        
         Company Name: [the name of the company that the esg report belongs to]
         Potential greenwashing claim: [the text's snippet of the potentian greenwashing claim]
         Label: [The type of greenwashing detected. For example, GREENWASHING TYPE 1]
         Justification: [short justification about the greenwashing type detected and how the claim fits that type]

         Another potential greenwashing claim: [another text snippet]
         Label: [The type of greenwashing detected. For example, GREENWASHING TYPE 1]
         Justification: [another justification]
         
         If no greenwashing claims are found, return this message:
         "No greenwashing claims found"

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         """


FIRST_PASS_PROMPT_WEB = """You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
        
         Greenwashing is defined as one of the following types
         
         - GREENWASHING TYPE 1:  using unreliable sustainability labels (e.g., H&M and Decathlon have used labels such as eco design and conscious
                                 and were prompted by the Netherlands Authority for Consumers and Markets to stop the practice in 2022 as the
                                 etailers did not clearly describe why the products received those labels).

         - GREENWASHING TYPE 2:  presenting legal requirements for the product as its distinctive features (e.g., McDonald’s advertises its reduction
                                 of plastic waste by using reusable cutlery, while this is a legal obligation since January 2023 in France)

         - GREENWASHING TYPE 3:  making green claims about the entire product when the claim applies only to a part/aspect of the product (e.g.,
                                 Kohl’s and Walmart were fined by the U.S. Federal Trade Commission for the misleading claim that their products
                                 were made of bamboo, a sustainable material, when they were made out of rayon, which is derived from bamboo
                                 via a highly toxic process that removes any environmental benefits of bamboo)

         - GREENWASHING TYPE 4:  making environmental claims for which the company cannot provide evidence (e.g., Keurig was fined by Canada’s
                                 Competition Bureau for misleading claims about the recycling process of its single-use coffee pods).      
                
         Your answer should follow the following format: 
        
         Company Name: [the name of the company that the esg report belongs to]
         Potential greenwashing claim raw: [The claim as you found it on the text] 
         Potential greenwashing claim: [A rewritten, specific, company-named sentence that can be queried.]
         Label: [The type of greenwashing detected. For example, GREENWASHING TYPE 1]
         Justification: [short justification about the greenwashing type detected and how the claim fits that type]

         Another potential greenwashing claim raw: [another claim as you found it on the text]    
         Another potential greenwashing claim: [another text snippet]
         Label: [The type of greenwashing detected. For example, GREENWASHING TYPE 1]
         Justification: [another justification]
         
         If no greenwashing claims are found, return this message:
         "No greenwashing claims found"

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         """


WEB_RAG_PROMPT = """You have at your disposal information '[Information]' and a statement: '[User Input]' whose accuracy must be evaluated. 
                            Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.

                            Before you decide:

                            1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                            2. Compare the statement with the information you have, evaluating each element of the statement separately.
                            3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                            Result: Provide a clear answer by choosing one of the following labels:

                            - NOT_GREENWASHING: If the statement is confirmed by the information and evidence.
                
                            - GREENWASHING: If the statement is a greenwashing statement based on the information and evidence.
                                                                        
                            - UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.

                            Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                            Statement: '[User Input]'
                            Result of the statement:
                            Justification:"""


CHROMA_PROMPT = """You have at your disposal information a statement: '[User Input]', extracted from a specific page: '[page_text]' of a report and relavant context: '[Context]' from the rest of the report, whose accuracy must be evaluated. 
                If part of information is missing, proceed with the analysis using only the information you have, or your knowledge.

                Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.

                Before you decide:

                1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                2. Compare the statement with the information from the rest of the report, evaluating each element of the statement separately.
                3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                Result: Provide a clear answer by choosing one of the following labels:

                - NOT_GREENWASHING: If the statement is confirmed by the information and evidence in the rest of the report.
                
                - GREENWASHING: If the statement is a greenwashing statement based on the information and evidence.
                                                               
                - UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.


                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                Statement: '[User Input]'
                            Result of the statement:
                            Justification:"""


LLM_AGGREGATOR_PROMPT = """You have at your disposal two independent verdicts regarding the accuracy of a given statement: '[User Input]'.  

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
                  
                - NOT_GREENWASHING: If the statement is confirmed by the information and evidence in the rest of the report.
                
                - GREENWASHING TYPE 1:  using unreliable sustainability labels (e.g., H&M and Decathlon have used labels such as eco design and conscious
                                        and were prompted by the Netherlands Authority for Consumers and Markets to stop the practice in 2022 as the
                                        retailers did not clearly describe why the products received those labels).

                - GREENWASHING TYPE 2:  presenting legal requirements for the product as its distinctive features (e.g., McDonald’s advertises its reduction
                                        of plastic waste by using reusable cutlery, while this is a legal obligation since January 2023 in France)

                - GREENWASHING TYPE 3:  making green claims about the entire product when the claim applies only to a part/aspect of the product (e.g.,
                                        Kohl’s and Walmart were fined by the U.S. Federal Trade Commission for the misleading claim that their products
                                        were made of bamboo, a sustainable material, when they were made out of rayon, which is derived from bamboo
                                        via a highly toxic process that removes any environmental benefits of bamboo)

                - GREENWASHING TYPE 4:  making environmental claims for which the company cannot provide evidence (e.g., Keurig was fined by Canada’s
                                        Competition Bureau for misleading claims about the recycling process of its single-use coffee pods). 
                                                               
                - UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.
                
                   
                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:
                
                Statement: '[User Input]'  
                Result of the statement:  
                Justification:"""

LLM_AGGREGATOR_PROMPT_2 = """You have at your disposal two independent verdicts regarding the accuracy of a given statement: '[User Input]'.  

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
                  
                - NOT_GREENWASHING: If the statement is confirmed by the information and evidence in the rest of the report.
                
                - GREENWASHING TYPE 1:  using unreliable sustainability labels (e.g., H&M and Decathlon have used labels such as eco design and conscious
                                        and were prompted by the Netherlands Authority for Consumers and Markets to stop the practice in 2022 as the
                                        retailers did not clearly describe why the products received those labels).

                - GREENWASHING TYPE 2:  presenting legal requirements for the product as its distinctive features (e.g., McDonald’s advertises its reduction
                                        of plastic waste by using reusable cutlery, while this is a legal obligation since January 2023 in France)

                - GREENWASHING TYPE 3:  making green claims about the entire product when the claim applies only to a part/aspect of the product (e.g.,
                                        Kohl’s and Walmart were fined by the U.S. Federal Trade Commission for the misleading claim that their products
                                        were made of bamboo, a sustainable material, when they were made out of rayon, which is derived from bamboo
                                        via a highly toxic process that removes any environmental benefits of bamboo)

                - GREENWASHING TYPE 4:  making environmental claims for which the company cannot provide evidence (e.g., Keurig was fined by Canada’s
                                        Competition Bureau for misleading claims about the recycling process of its single-use coffee pods). 
                                                               
                - UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.
                

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:
                
                Statement: '[User Input]'  
                Result of the statement:  
                Justification:"""


LLM_AGGREGATOR_PROMPT_FINAL = """You have at your disposal multiple independent verdicts regarding the accuracy of a given statement: '[User Input]'.  

                - The first verdict (Web Verdict) is derived from an LLM that retrieves and analyzes information from the web to assess the claim. If no content was found from the web, the Web Verdict will contain 'No content was found from the web'. Proceed with the analysis using only the Database Verdict.
                - The second verdict (Reddit Verdict) is derived from an LLM that analyzes user-generated discussions and opinions related to the statement. If no content was found from Reddit, the Reddit Verdict will contain 'No content was found from Reddit'. Proceed with the analysis using only the Database Verdict.
                - The third verdict (Chroma Verdict) is derived from an LLM that has access to a structured database containing the entire document from which the statement was extracted. If no content was found from Chroma, the Chroma Verdict will contain 'No content was found from Chroma'. Proceed with the analysis using only the Database Verdict.
                - The fourth verdict (News Verdict) is derived from an LLM that analyzes news articles related to the statement. If no content was found from News, the News Verdict will contain 'No content was found from News'. Proceed with the analysis using only the Database Verdict.
                - The fifth verdict (Chroma ESG Verdict) is derived from an LLM that has access to a structured database containing the entire document from which the statement was extracted. If no content was found from Chroma ESG, the Chroma ESG Verdict will contain 'No content was found from Chroma ESG'. Proceed with the analysis using only the Database Verdict.
                
                Your task is to analyze all the verdicts and reach a final conclusion regarding the statement's accuracy.  

                Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.  

                Before making your final decision:  

                1. Analyze the given statement clearly to identify its key elements.  
                2. Examine the reasoning in both the Database Verdict and the Web Verdict.  
                3. Compare both verdicts and resolve any discrepancies by determining which source provides stronger, more reliable justification.  
                4. Use your own reasoning to synthesize the evidence and reach a final, well-supported conclusion.  

                Possible Results:  
                  
                - NOT_GREENWASHING: If the statement is confirmed by the information and evidence in the rest of the report.
                
                - GREENWASING: If the statement is a greenwashing claim based on the evidence.
                                                               
                - UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:
                
                Statement: '[User Input]'  
                Result of the statement:  
                Justification:"""

REDDIT_PROMPT = """You have at your disposal information a statement: '[User Input]', extracted from a specific page: '[page_text]' of a report and relavant context: '[Context]' from a database with reddit posts, from a greenwashing subreddit, whose accuracy must be evaluated. 
                If part of information is missing, proceed with the analysis using only the information you have, or your knowledge.

                Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.

                Before you decide:

                1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                2. Compare the statement with the information from the rest of the report, evaluating each element of the statement separately.
                3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                Result: Provide a clear answer by choosing one of the following labels:

                - NOT_GREENWASHING: If the statement is confirmed by the information and evidence in the rest of the report.
                
                - GREENWASHING TYPE 1:  using unreliable sustainability labels (e.g., H&M and Decathlon have used labels such as eco design and conscious
                                        and were prompted by the Netherlands Authority for Consumers and Markets to stop the practice in 2022 as the
                                        retailers did not clearly describe why the products received those labels).

                - GREENWASHING TYPE 2:  presenting legal requirements for the product as its distinctive features (e.g., McDonald’s advertises its reduction
                                        of plastic waste by using reusable cutlery, while this is a legal obligation since January 2023 in France)

                - GREENWASHING TYPE 3:  making green claims about the entire product when the claim applies only to a part/aspect of the product (e.g.,
                                        Kohl’s and Walmart were fined by the U.S. Federal Trade Commission for the misleading claim that their products
                                        were made of bamboo, a sustainable material, when they were made out of rayon, which is derived from bamboo
                                        via a highly toxic process that removes any environmental benefits of bamboo)

                - GREENWASHING TYPE 4:  making environmental claims for which the company cannot provide evidence (e.g., Keurig was fined by Canada’s
                                        Competition Bureau for misleading claims about the recycling process of its single-use coffee pods). 
                                                               
                - UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                Statement: '[User Input]'
                            Result of the statement:
                            Justification:"""

# TODO: add a prompt for the chroma_esg aggregator
SOLO_AGGREGATOR_PROMPT = """You have at your disposal three independent sources of context regarding the accuracy of the following statement: '[User Input]'.

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
- UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.

Finally, explain your reasoning clearly. Focus on the provided data and your own critical evaluation. Avoid unnecessary details and aim for precision and clarity in your analysis. Use the following format:

Statement: '[User Input]'  
Result of the statement:  
Justification:"""

CHROMA_ESG_PROMPT = """You have at your disposal information a statement: '[User Input]', extracted from a specific page: '[page_text]' of a report and relavant context: '[Context]' from a database with past esg reports, regarding the same company. 
                If part of information is missing, proceed with the analysis using only the information you have, or your knowledge.

                Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.

                Before you decide:

                1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                2. Compare the statement with the information from the rest of the report, evaluating each element of the statement separately.
                3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                Result: Provide a clear answer by choosing one of the following labels:

                - NOT_GREENWASHING: If the statement is confirmed by the information and evidence in the rest of the report.
                
                - GREENWASHING TYPE 1:  using unreliable sustainability labels (e.g., H&M and Decathlon have used labels such as eco design and conscious
                                        and were prompted by the Netherlands Authority for Consumers and Markets to stop the practice in 2022 as the
                                        retailers did not clearly describe why the products received those labels).

                - GREENWASHING TYPE 2:  presenting legal requirements for the product as its distinctive features (e.g., McDonald’s advertises its reduction
                                        of plastic waste by using reusable cutlery, while this is a legal obligation since January 2023 in France)

                - GREENWASHING TYPE 3:  making green claims about the entire product when the claim applies only to a part/aspect of the product (e.g.,
                                        Kohl’s and Walmart were fined by the U.S. Federal Trade Commission for the misleading claim that their products
                                        were made of bamboo, a sustainable material, when they were made out of rayon, which is derived from bamboo
                                        via a highly toxic process that removes any environmental benefits of bamboo)

                - GREENWASHING TYPE 4:  making environmental claims for which the company cannot provide evidence (e.g., Keurig was fined by Canada’s
                                        Competition Bureau for misleading claims about the recycling process of its single-use coffee pods). 
                                                               
                - UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.

                Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                Statement: '[User Input]'
                            Result of the statement:
                            Justification:"""

NEWS_PROMPT = """You have at your disposal information '[Information]' and a statement: '[User Input]' whose accuracy must be evaluated.
                            Use only the provided information in combination with your knowledge to decide whether the statement is GREENWASHING or NOT_GREENWASHING.
                            The information is gathered from various news sites regarding sustainabillity news, news regarding the esg standards, greenwashing news and news about the climate.

                                                       
                            Before you decide:

                            1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                            2. Compare the statement with the information you have, evaluating each element of the statement separately.
                            3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.
                            4. The statement should be in a form of a sentence of a sub-sentence and not a title or a question.

                            Result: Provide a clear answer by choosing one of the following labels:

                             - NOT_GREENWASHING: If the statement is confirmed by the information and evidence.
                
                             - GREENWASING: If the statement is a greenwashing claim based on the evidence.
                                                               
                             - UNCLEAR: If the sources do not provide a clear conclusion regarding the statement's accuracy.
                            
                            Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. 
                            Avoid unnecessary details and try to be precise and concise in your analysis. 
                            Take the publish date under consideration. If the article is old, it might not be relevant anymore.
                            If the article is recent, it might be more relevant.
                            Your answers should be in the following format:

                            Statement: '[User Input]'
                            Result of the statement:
                            Justification:"""


text = """
                            Use the GREENWASHING label if at least one of the following is true:

                            1. using unreliable sustainability labels (e.g., H&M and Decathlon have used labels such as eco design and conscious
                            and were prompted by the Netherlands Authority for Consumers and Markets to stop the practice in 2022 as the
                            retailers did not clearly describe why the products received those labels)
                            2. presenting legal requirements for the product as its distinctive features (e.g., McDonald’s advertises its reduction
                            of plastic waste by using reusable cutlery, while this is a legal obligation since January 2023 in France)
                            3. making green claims about the entire product when the claim applies only to a part/aspect of the product (e.g.,
                            Kohl’s and Walmart were fined by the U.S. Federal Trade Commission for the misleading claim that their products
                            were made of bamboo, a sustainable material, when they were made out of rayon, which is derived from bamboo
                            via a highly toxic process that removes any environmental benefits of bamboo), and finally
                            4. making environmental claims for which the company cannot provide evidence (e.g., Keurig was fined by Canada’s
                            Competition Bureau for misleading claims about the recycling process of its single-use coffee pods)."""
