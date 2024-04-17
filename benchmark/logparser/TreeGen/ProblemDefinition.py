import numpy as np
import pandas as pd
import random
import regex as re
import os

from nltk.tokenize import TreebankWordTokenizer

import matplotlib.pyplot as plt
import networkx as nx

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np

class GAProblem():
    def __init__(self,log_file,output_dir,rex):
        self.output_dir=output_dir
        self.log_file=log_file
        self.rex=rex
    
        
        NOBJ=2
        self.ref_points=tools.uniform_reference_points(NOBJ, 4)
        
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)


        self.toolbox = base.Toolbox()
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual,self.get_layers)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.eval_individual)
        
        self.toolbox.register("select",self.select)
        self.toolbox.register("mate", self.cross_over)
        self.toolbox.register("mutate", self.mutate)
        
       

    def eval_individual(self,individual):
        
        visit_rate=[0]*self.layers_count
        nodes_count=[0]*self.layers_count
        #print(visit_rate)
        for tokens in self.tokenized:
                       
            for token in tokens:
                layer_pos=token[0]
                
                if self.is_static(individual,token):
                    visit_rate[layer_pos]+=1
                
                if nodes_count[layer_pos]==0:
                    nodes_count[layer_pos]=len(individual[layer_pos])
            
        # print(visit_rate,nodes_count)
        
        weighted_average_rate = sum(visit_rate[i] * len(self.knowledge_base[i]) for i in range(self.layers_count)) / self.weighted_sum
        
        weighted_nodes_count = sum(nodes_count[i] * len(self.knowledge_base[i]) for i in range(self.layers_count)) / self.weighted_sum
        average_visit_rate = sum(visit_rate) / len(visit_rate)
        average_nodes_count = sum(nodes_count) / len(nodes_count)
        # print("Average Visit Rate:", average_visit_rate)
        # print("Average Nodes Count:", average_nodes_count)
        return weighted_average_rate, weighted_nodes_count
    
    def prep_KB(self,train_data):
        self.tokenized, self.data_delimiters = zip(*train_data.map(self.tokenizer))
        self.knowledge_base,self.adj_vector=self.get_knowledge_base()
        self.layers_count=len(self.knowledge_base)
        self.weighted_sum=sum(len(v) for v in self.knowledge_base.values())

    def train(self,GA_params):
        
        NPOP=GA_params["NPOP"]
        NGEN=GA_params["NGEN"]
        SEED=GA_params["SEED"]
        CXPB=GA_params["CXPB"]
        MUTPB=GA_params["MUTPB"]
        
        random.seed(SEED)
        
        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = self.toolbox.population(n=NPOP)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(1, NGEN):
            offspring = algorithms.varAnd(pop, self.toolbox, CXPB, MUTPB)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            pop = self.toolbox.select(pop + offspring, NPOP)

            # Compile statistics about the new population
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
        
        return pop,logbook
    
    def extract_graph(self,pop):
        graph=[]*self.layers_count
        for tokens in self.tokenized:
                     
            for token in tokens:
                
                evals=[]
                for ind in pop:

                   evals.append(self.is_static(ind,token))
                
        return graph
    
    def get_templates(self,inds,data):
        
        tokenized, data_delimiters = zip(*data.map(self.tokenizer))

        templates=[]
        parameters=[]
        i=1
        for tokens, delimiters, log in zip(tokenized,data_delimiters,data):
            template=[]
            template_params=[]
           
            for token in tokens:
                delimiter=delimiters[token[0]] if token[0] < len(delimiters) else '' 
                
                evals=[]
                for ind in inds:

                   evals.append(self.is_static(ind,token))
                      
                final_token = token[1]+delimiter if sum(evals) > len(evals)*0.1 else '<*>'+delimiter
                        
                template.append(final_token)
                template_params.append(token[1])
                    
              
               
            if i==1914-1:
                print(log)
                print(tokens)
                print(delimiters)
                print(''.join(template),"\n")
            i+=1
            parameters.append(template_params)
            templates.append(''.join(template))
            
        # add ID
        templates=[(template,str(params)) for template,params in zip(templates,parameters)]
        df=pd.DataFrame(templates,columns=['Template',"Params"])
        # print(df)
        return df
    
    def is_static(self,ind,token):
        token_pos=token[0]
        token=token[1]
        
        # # check if key dict does not exist
        # if (token_pos not in self.knowledge_base) or (token_pos >= len(ind)):
        #     return 0
    
        # find token in layer
        token_id = self.knowledge_base[token_pos].index(token) if token in self.knowledge_base[token_pos] else None
            
        #print(token,layer,token_pos, token_id)
        if token_id in ind[token_pos]:
            return 1
      
        return 0
        
       
        
    
    def plot_population(self,pop,name):
        
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        p = np.array([ind.fitness.values for ind in pop])
        ax.scatter(p[:, 0], p[:, 1], marker="o", s=24, label="Final Population")
        
        ax.autoscale(tight=True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, self.log_file + '_'+name+".png"))
        
    def cross_over(self,ind1,ind2):
        child1, child2 = self.toolbox.clone(ind1),self.toolbox.clone(ind2)
      
        for l1,l2,l_n in zip(ind1,ind2,range(len(ind1))):
            # print("old_cx",l1,l2)
            if l_n==0:
                tools.cxUniform(l1,l2, indpb=0.5)
                child1[l_n]=np.unique(np.append(l1,-1).astype(int))
                child2[l_n]=np.unique(np.append(l2,-1).astype(int))
            else:
                l1_succ=self.get_successors(child1[l_n-1],l_n)
                l2_succ=self.get_successors(child2[l_n-1],l_n)
                l1=np.append([node for node in l1 if node in l1_succ],-1).astype(int)
                l2=np.append([node for node in l2 if node in l2_succ],-1).astype(int)
                #tools.cxUniform(l1,l2, indpb=0.5)
                child1[l_n]=np.unique(np.append(l1,-1).astype(int))
                child2[l_n]=np.unique(np.append(l2,-1).astype(int))
            # print("new_cx",child1[l_n],child2[l_n])
        # add -1 to layer
                
                                  
       
        return (child1,child2)
   
    # modify list inside loop example
    
    
    
    
    def mutate(self,ind):
        # clone individual
        mutant = self.toolbox.clone(ind)
        for l,i in zip(ind,range(len(ind))):
            # print("old",l)
            if i == 0:
                low,up=-1,len(self.knowledge_base[i])-1
                l=tools.mutUniformInt(l,low,up,indpb=0.2)
                mutant[i]=np.unique(np.append(l,-1).astype(int))
            else:
                l_succ=self.get_successors(mutant[i-1],i)
                l=[node for node in l if node in l_succ]
                #space to mutation     
                #l = np.where(np.random.rand(len(l)) < 0.2, np.random.choice(l_succ, len(l),replace=False), l)
                mutant[i]=np.unique(np.append(l,-1).astype(int))
            # print("new",mutant[i])
           
            
            
            # print("new",l)
            
        return mutant,
        
    
    def select(self,inds,k):
        return tools.selNSGA3(inds,k,ref_points=self.ref_points)
    
    def get_successors(self,prev_layer,i):
        possible_succesors=[]
        for node_index in prev_layer:
            if node_index==-1:
                continue
            adj_key=self.knowledge_base[i-1][node_index]
            possible_succesors.extend(self.adj_vector[adj_key])
        return [index for index,node in enumerate(self.knowledge_base[i]) if np.isin(node,possible_succesors)]

    def get_layer(self,max_nodes,prev_layer,i):
        nodes_cnt=random.randint(0,max_nodes)
        if prev_layer is None:
            # init array of unique random numbers in interval
            layer=np.random.choice(max_nodes, nodes_cnt, replace=False)
        else:
            possible_succesors=self.get_successors(prev_layer,i)
            layer=np.random.choice(possible_succesors, nodes_cnt, replace=False) if nodes_cnt <= len(possible_succesors) else possible_succesors
        # add -1 to layer
        layer=np.append(layer,-1).astype(int)
        # if max_nodes==0:
        #     print(layer)
        return np.sort(layer)
    
    def get_layers(self):
        layers = [0]*self.layers_count
        for i in range(self.layers_count):
            layers[i]=self.get_layer(len(self.knowledge_base[i]), prev_layer=None if i==0 else layers[i-1],i=i)
        
        return layers
    
    def tokenizer(self, line):
        # print("old",line)
        line=line.strip()
        filter_patterns=[#("time",'(^|\s+)(\d){1,2}:(\d){1,2}(|:(\d){2,4})(\s+|$)'),
                         #("date",'(^|\s)(\d{1,2}(-|/)\d{1,2}(-|/)\d{2,4})(\s|$)'),
                         ("ip",'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)'),
                         ("ip1",'(|^)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(|$)'),
                         ("file_path",'(/[\d+\w+\-_\.\#\$]*[/\.][\d+\w+\-_\.\#\$/*]*)+'),
                         ("memory_addr",'0x([a-zA-Z]|[0-9])+'),
                         ("hex",'(^|\s)([0-9A-F]){9,}(\s|$)'),
                         ("hex1",'(^|\s)([0-9a-f]){8,}(\s|$)'),
                         ("mac",'([0-9A-F]{2}[:-]){5,}([0-9A-F]{2})')
                         #("number",'(^| )\d+( |$)')
                         ]
        for filter_pattern in filter_patterns:
            if filter_pattern[0] in ["hex","hex1"]:
                line = re.sub(filter_pattern[1], ' 0 ', line) 
            line = re.sub(filter_pattern[1], '0', line)
        # print("patterns",line)
        # filter some basic elements like ip, digits, and so on
        for rex in self.rex:
            line = re.sub(rex, '0', line)
        # print(line)
        tokenizer_style="basic"  
        if tokenizer_style=="model":
            tokens=TreebankWordTokenizer().tokenize(line)
            # print(tokens)
        else:
            for bracket in ["()","[]","{}"]:
                line = re.sub(rf'({re.escape(bracket)})', bracket[0]+'0'+bracket[1], line)
            # basic_delimiters = r'(\s=\s|\s|,|;|=|#+|:)'
            base_chars_with_spaces=r'(\.|\-|\#|\$|\@|\(|\)|\{|\}|\[|\]|\<|\>|\=|\+|\-|\*|\/|\%|\^|\&|\!|\?|\:|\;|\,|\"|\'|\`|\~|\||\s)+'#\.\-
                 
            pattern=re.compile(base_chars_with_spaces)

            delimiters=[match.group() for match in pattern.finditer(line)]
          
                
            if delimiters==[]:
                tokens=[line]
            else:
                tokens = re.split('|'.join(map(re.escape, sorted(delimiters, key=len, reverse=True))), line)
            # print(tokens)
            # sort list by length

            months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            # concatenate list
            date_tokens=months+days
            tokens=[token if token not in date_tokens else '0' for token in tokens]
            
            # add positon of token in line
            # if '' in tokens:
            #     print(line,tokens,delimiters)
            tokens=[token for token in tokens if token != '']
           
            all_tokens=[(index, token, tokens[index+1] if len(tokens)!= index+1 else '0') for index, token in enumerate(tokens)]
            # withou floats, ints, <*>
            # tokens_nodigits= [(index, token) for index, token in enumerate(tokens) if not re.match('(^| )\d+( |$)',token)]
            # print(all_tokens, tokens_nopattern,"\n")
            return all_tokens, delimiters
        

    def get_knowledge_base(self):
        
        # map over series
       
        # flatten list of lists and make it unique   
        
        
        unique_tokens = np.unique(np.concatenate(self.tokenized),axis=0)
        #print(unique_tokens)
        # dictonary of key and its tokens
        knowledge_base = {}
        adj_vector={}
        for token in unique_tokens:
            key = int(token[0])
            if key not in knowledge_base:
                knowledge_base[key]=[]
            
                
            if not re.match('(^| )\d+( |$)',token[1]):
                if token[1] not in knowledge_base[key]:
                    knowledge_base[key].append(token[1])
              
        
            # adjacency vector
            if token[1] not in adj_vector:
                adj_vector[token[1]]=[]
            
            if not re.match('(^| )\d+( |$)',token[2]):
                if token[2] not in adj_vector[token[1]]:
                    adj_vector[token[1]].append(token[2])
            
        # size of dictionary
        #print(knowledge_base[0])
        print({k:len(v) for k, v in knowledge_base.items()})
        print("Knowledge Base Size:",len(knowledge_base))
        
        
        
        return knowledge_base,adj_vector
    
   