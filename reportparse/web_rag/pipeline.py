import os
from reportparse.web_rag.search import google_search
from reportparse.web_rag.harvest import Harvester
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class pipeline:

    def __init__(self, query, n, metadata):
        self.query = query
        self.n = n
        self.harvested_urls= None
        self.meta = metadata

    #based on a claim, implement a searcher and a harvester 
    def retrieve_knowledge(self):

        #scan the web for urls containing knowledge
        url_list = google_search(self.query, self.n+2, self.meta)
        if url_list is None:
            print('Could not find any results regarding the claim. Please try again or choose a different statement')
            return None
        else:
            #harvest the external urls using a harvester instance
            my_harvester = Harvester(list(url_list), self.query, timeout=1000, claim_id=0, max_sources = self.n)
            df = my_harvester.run()
            #df.to_csv('result_df.csv', index=False)
            #get the bodies of the top-n web sources that has the biggest "body_similarity" value
            try:
                result = df.nlargest(self.n, 'body_similarity')['similar_sentences'] 
                self.harvested_urls = df.nlargest(self.n, 'body_similarity')['url'].to_list()
            except Exception as e:
                print('Could not find relevant sources.')
                return None, []
        
        return result, self.harvested_urls
    

# pip = pipeline('Toyota factory is harming the enviroment', 3)
# result = pip.retrieve_knowledge()
# print(print("\n".join(result.astype(str))) )
