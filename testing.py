
from langchain_groq import ChatGroq
import os
import time


def call_llm(text): 
       
        
        time.sleep(5)

        llm = ChatGroq( 
            model=os.getenv("GROQ_LLM_MODEL_2"), 
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            groq_api_key=os.getenv("GROQ_API_KEY_1"),
        )

        llm_2 = ChatGroq(
            model=os.getenv("GROQ_LLM_MODEL_1"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            groq_api_key=os.getenv("GROQ_API_KEY_2"),
        )


        messages = [
            (
                "system",
                f'''You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 

         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Another potential greenwashing claim: [another claim]
         Justification for the second claim: [another justification]
         
         If no greenwashing claims are found, return this message:
         "No greenwashing claims found"

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         ''',
                
            ),
            ("human", f"{text}"),
        ]

        try:
            print('Invoking with the first llm...')
            if len(text.split()) >= 19:
                ai_msg = reduce_llm_input(text, llm)
                return ai_msg
            else:
                ai_msg = llm.invoke(messages)
                return ai_msg.content
        except Exception as e:
            print(e)
            try:
                print('Invoking with the second llm...')
                if len(text.split()) >= 1900:
                    ai_msg = reduce_llm_input(text, llm_2)
                    return ai_msg
                else:
                    ai_msg = llm.invoke(messages)
                    return ai_msg.content
            except Exception as e:
                print('LLM invokation failed. Returning none...')
                print(e)
                return None
            

def reduce_llm_input(text, llm):
        
        print('Invoking map reduce function to split text')
        from langchain.prompts import PromptTemplate
        from langchain.chains.summarize import load_summarize_chain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        

        #Dynamically split the text into three parts
        chunk_size = len(text) // 3  
        chunk_1 = text[:chunk_size]
        chunk_2 = text[chunk_size:2*chunk_size]
        chunk_3 = text[2*chunk_size:]  


        

        # Define the map template
        map_template = """You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 

         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Second potential greenwashing claim: [another claim]
         Justification for the second claim: [another justification]

         Third potential greenwashing claim: [another claim]
         Justification for the third claim: [another justification]
         
         If no greenwashing claims are found, return nothing.
         
         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
        Text to be examined: {docs}"""
        map_prompt = PromptTemplate.from_template(map_template)

       # Define the reduce template
        reduce_template = """Synthesize the following results, into a single conlcusion. Please follow the format that is given to you.
                            If no greenwashing claims are found, return this message:
                            "No greenwashing claims found"
                            If greenwashing claims were found, follow the format below:
                            Potential greenwashing claim: [the claim]
                            Justification: [short justification]

                            Second potential greenwashing claim: [another claim]
                            Justification for the second claim: [another justification]

                            Third potential greenwashing claim: [another claim]
                            Justification for the third claim: [another justification]

                            Do not make any commentary and don't create any titles. Just provide what you are told.
                           The result are listed below: {docs}"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        # Use the new syntax for creating the map and reduce steps
        map_chain = map_prompt | llm  # Chain map prompt with LLM
        reduce_chain = reduce_prompt | llm  # Chain reduce prompt with LLM

        # Process each chunk with the map_chain (i.e., invoke the LLM three times)
        result_1 = map_chain.invoke({"docs": chunk_1})
        time.sleep(5)
        result_2 = map_chain.invoke({"docs": chunk_2})
        time.sleep(5)
        result_3 = map_chain.invoke({"docs": chunk_3})
        time.sleep(5)
        

        # Combine the results into one final summary
        
        result_1_text = result_1.content if hasattr(result_1, 'content') else str(result_1)
        result_2_text = result_2.content if hasattr(result_2, 'content') else str(result_2)
        result_3_text = result_3.content if hasattr(result_3, 'content') else str(result_3)

        # Use the reduce_chain to generate the final summary from the combined results
        
        combined_results = "\n".join([result_1_text, result_2_text, result_3_text])
       
        final_summary = reduce_chain.invoke({"docs": combined_results})

        
        #print(final_summary.content if hasattr(final_summary, 'content') else str(final_summary))
        result = final_summary.content if hasattr(final_summary, 'content') else str(final_summary)

        return result


text = '''Honeywell can deliver solutions to help drive the energy
transition and decarbonization. We have unique expertise in
essential technologies needed to help on the journey to create
a net-zero economy, including refrigerants, renewable diesel
and aviation fuels, hydrogen production, and carbon capture,
utilization and storage (CCUS).
SOLSTICE PRODUCTS HELP REDUCE CO2 EMISSIONS
Honeywell Solstice® technology helps deliver on pledges to
reduce carbon emissions. Use of Honeywell Solstice products
has helped avoid the potential release of the equivalent of
more than 326 million metric tons of carbon dioxide into the
atmosphere, comparable to the CO2 sequestered by 389 million
acres of U.S. forests for one year.1 The numbers keep rising
every day.
“Whether they are used to air condition cars, deliver medicine
via an inhaler, propel home or personal care products or produce
foam insulation, Solstice touches the lives of millions of people
every day,” says Jeff Dormo, President, Honeywell Advanced
Materials. “Honeywell has invested more than $1 billion in
research, development and new capacity for Solstice technology
over the last decade to create products that help customers
reduce their carbon footprint without sacrificing end-product
performance.”
Solstice products use breakthrough hydrofluoroolefin (HFO)
technology, the most effective alternative to conventional
hydrofluorocarbons (HFCs), which are being phased out
because they have very high global warming potential (GWP). In
fact, HFOs typically have GWPs that are more than 99% lower
than the equivalent HFCs.
RENEWABLE FUELS HELP DRIVE EFFORTS TO REACH
TO NET ZERO
Producing high-performance, low-emissions renewable diesel
and sustainable aviation fuel (SAF) is nothing new for Honeywell
UOP. We pioneered the UOP Ecofining™ technology more
than a decade ago to produce sustainable fuels from waste
feedstocks like inedible fats, oils and greases. Since then, we
have developed innovative new processes to expand options for
renewable fuel refiners and end users.
“Our latest Ecofining™ technology is a ready-now and efficient
way refiners can produce renewable diesel and SAF today. Both
fuels are nearly chemically identical to their petroleum-based
counterparts and are used as drop-in replacements without
modification to engines or fuel systems,” said Kevin O’Neil,
Senior Business Leader, Renewable Fuels. “Depending on the
feedstock used, diesel and SAF produced from the Ecofining™
process can reduce GHG emissions by 60 to 80 percent on a
total lifecycle basis, compared to petroleum-based fuels.”2
In addition to Ecofining™, Honeywell has introduced new
ethanol to jet (ETJ) and methanol to jet (MTJ) processes to
produce SAF from other readily available feedstocks. The ETJ
process uses corn-based cellulosic or sugar-based ethanol. Our
newest ready-now MTJ technology converts eMethanol to eSAF.'''

print(call_llm(text))