
import argparse
import os
import pickle
import sys
import pandas as pd
import numpy as np
import hashlib
from ..utils import logloader
from datetime import datetime
import re
from .ProblemDefinition import GAProblem


class LogParser():
    def __init__(self, indir, outdir, log_format, GA,  rex=[], n_workers=1):
        self.input_dir = indir
        self.output_dir = outdir
        self.log_format = log_format
        self.rex = rex
        self.n_workers = n_workers
        self.templates = []
        self.GA_params=GA

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
            
    
    
    
    def init_GA(self,log_file):
        self.problem=GAProblem(log_file,self.output_dir,self.rex)
        return
        
    def train(self,train_data):
        # ind1=self.problem.toolbox.individual()
        # ind2=self.problem.toolbox.individual()
        # print(ind1)
        # test mutate, mate, selecet
       
        # ind1,ind2=self.problem.toolbox.mate(ind1,ind2)
        # print(ind1,ind2)
        
        # ind1=self.problem.toolbox.mutate(ind1)
        # print(ind1)
        #population=self.problem.toolbox.population(n=100)
        # selection=self.problem.toolbox.select(population,5)
        #print(selection)
        #self.problem.eval_individual(ind1)
        
        self.problem.prep_KB(train_data)
        
        # Concatenate values of dictionary
        checkpoint_file = "./tmp/"+self.problem.log_file
        for key, value in self.GA_params.items():
            checkpoint_file += "-"+str(value)
        print(checkpoint_file)
        
        try:
            with open(checkpoint_file, "rb") as cp_file:
                cp = pickle.load(cp_file)
            print("Loaded Checkpoint")
            pop = cp["population"]
            stats = cp["logbook"]
        except:
            pop,stats=self.problem.train(self.GA_params)
        
        cp = dict(population=pop,logbook=stats)
        with open(checkpoint_file, "wb") as cp_file:
                pickle.dump(cp, cp_file)
        
        self.problem.extract_graph(pop)
        #print(pop)
        #print("LOGBOOK",stats)
        # dumb individual
        pickle.dump(list(pop[len(pop)//2]), open("./tmp/pop"+self.problem.log_file+".pkl", "wb"))
        self.problem.plot_population(pop,'nsga3')
        
        return pop
    
    def extract_train_data(self,log_data):
        data=pd.Series(log_data.unique())
        print("Reduced data:",data.shape)
        return data
    
        
    def extract_templates(self,pop,data):
        # best_n=self.problem.pareto_front(pop,n=5)
        # choose the best individual with min obj1\
        # avg=int(len(best_n)*self.GA_params["Q"])
        # best=best_n[avg]
        templates=self.problem.get_templates(pop,data)
        return templates
        
    def parse(self, log_file):
        starttime = datetime.now() 
        loader = logloader.LogLoader(self.log_format, self.n_workers)
        log_dataframe = loader.load_to_dataframe(os.path.join(self.input_dir, log_file))
        print(log_dataframe.head())
        
        train_data=self.extract_train_data(log_dataframe["Content"])
        self.init_GA(log_file)
        pop=self.train(train_data)
        templates=self.extract_templates(pop,log_dataframe["Content"])

        
        log_dataframe['EventTemplate'] = templates["Template"]
        log_dataframe['EventId'] = templates["Template"].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])

        # get unique values and its counts
        unique_templates = templates["Template"].value_counts().reset_index()
        #print(unique_templates)
        
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = unique_templates['Template']
        df_event['Occurrences'] = unique_templates["count"]
        df_event['EventId'] = unique_templates['Template'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event = df_event.sort_values('EventTemplate', ascending=True).reset_index(drop=True)
        df_event.to_csv(os.path.join(self.output_dir, log_file + '_templates.csv'), index=False, columns=["EventId", "EventTemplate", "Occurrences"])
        log_dataframe.to_csv(os.path.join(self.output_dir, log_file + '_structured.csv'), index=False)
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))
        

