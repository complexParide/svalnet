# importing necessary modules
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import scipy.stats as st
from itertools import combinations
from multiprocessing import Pool, cpu_count


# main class of the module
class SVN():
    
    def __init__(self, edgelist: pd.DataFrame, proj_column, nonproj_column):
       	# edgelist: pandas dataframe of the edgelist
       	# proj_column: name of the column of nodes we want to project and validate
       	# nonproj_column: name of the column of nodes we don't want to project
       	# All the other columns in the edgelist will be ignored
        self.proj_nodes = np.unique(self.edgelist[proj_column])
        self.nonproj_nodes = np.unique(self.edgelist[nonproj_column])
        self.N = self.proj_nodes.shape[0]
        self.Nn = self.nonproj_nodes.shape[0]
        self.proj_to_idx = dict(zip(self.proj_nodes, range(self.N)))
        self.nonproj_to_idx = dict(zip(self.nonproj_nodes, range(self.Nn)))
        self.idx_to_proj = dict(zip(range(self.N), self.proj_nodes))
        self.idx_to_nonproj = dict(zip(range(self.Nn), self.nonproj_nodes))
        print(f"Generating biadjacency and projected matrices...")
        self.A = np.zeros((self.N, self.Nn))
        for _, line in self.edgelist.iterrows():
            self.A[self.proj_to_idx[line[proj_column]]][self.nonproj_to_idx[line[nonproj_column]]]=1
        self.P = np.matmul(self.A, self.A.T) * (np.ones((self.N, self.N)) - np.identity(self.N))
        
    def oeFunc(self, ij):
    	# helper function to calculate over-expressions
        return [*ij, st.hypergeom.sf(self.P[*ij] - 1, self.Nn,
                                     self.A[ij[0]].sum(), self.A[ij[1]].sum())]
    
    def ueFunc(self, ij):
    	# helper function to calculate under-expressions
        return [*ij, st.hypergeom.cdf(self.P[*ij], self.Nn,
                                      self.A[ij[0]].sum(), self.A[ij[1]].sum())]
        
    def overExpression(self, threshold=0.01, ncores=1,
                       bon_save_path=None, fdr_save_path=None, sep=','):
        # Function that calculates over-expressions and saves them as edgelists of the SVN
        # threshold: arbitrary validation threshold
        # ncores: number of cores used to parallelize the project. If 1 there is no parallelization
        # bon_save_path: path where to save Bonferroni SVN. If None it's not saved
        # bon_save_path: path where to save FDR SVN. If None it's not saved
        # sep: separator of the saved edgelists
        nonzeros = np.nonzero(np.triu(self.P))
        edge_iterator = zip(*nonzeros)
        if ncores == 1:
            table = []
            for ab in tqdm(edge_iterator, total=nonzeros[0].shape[0], colour='green',
                           ncols=111, desc='Calculating OE p-values'):
                table.append(self.oeFunc(ab))
        else:
            print(f"Calculating OE p-values (this might take a while)...")
            start = time.time()
            ncores = np.min((ncores, cpu_count()))
            dead = Pool(processes=ncores)
            table = dead.map(self.oeFunc, edge_iterator)
            print(f"OE probabilities ready [tte = {(time.time() - start) // 60} minutes]")
        df = pd.DataFrame(table, columns=['source', 'target', 'pvalue'])
        df['source'] = df['source'].replace(self.idx_to_proj)
        df['target'] = df['target'].replace(self.idx_to_proj)
        t = np.arange(1, df.shape[0] + 1) * threshold * 2 / (self.N * (self.N - 1))
        df = df.sort_values('pvalue')
        fdr_df = df[df['pvalue'] < t]
        fdr_df = fdr_df.assign(threshold=t[:fdr_df.shape[0]])
        if fdr_save_path is not None:
            fdr_df.to_csv(fdr_save_path, sep=sep, index=None)
        print(f"FDR applied! {fdr_df.shape[0]} links have been validated!")
        t = threshold * 2 / (self.N * (self.N - 1))
        bon_df = fdr_df[fdr_df['pvalue'] < t]
        bon_df = bon_df.assign(threshold=np.ones(bon_df.shape[0]) * t)
        if bon_save_path is not None:
            bon_df.to_csv(bon_save_path, sep=sep, index=None)
        print(f"Bonferroni applied! {bon_df.shape[0]} links have been validated!")
        return fdr_df, bon_df
        
    def underExpression(self, threshold=0.01, ncores=1, batch_size=500000,
                        bon_save_path=None, fdr_save_path=None, sep=','):
        # Function that calculates over-expressions and saves them as edgelists of the SVN
        # threshold: arbitrary validation threshold
        # ncores: number of cores used to parallelize the project. If 1 there is no parallelization
        # batch_size: splits the parallelization if ncores>1
        # bon_save_path: path where to save Bonferroni SVN. If None it's not saved
        # bon_save_path: path where to save FDR SVN. If None it's not saved
        # sep: separator of the saved edgelists
        combs = self.N * (self.N - 1) // 2
        edge_iterator = combinations(range(self.P.shape[0]), 2)
        if ncores == 1:
            table = []
            for ab in tqdm(edge_iterator, total=combs, colour='green',
                           ncols=111, desc='Calculating UE p-values'):
                line = self.ueFunc(ab)
                if line[-1] < threshold:
                    table.append(line)
        else:
            start = time.time()
            ncores = np.min((ncores, cpu_count()))
            table = []
            bn = 0
            for batch in tqdm(self.batchIterator(edge_iterator, batch_size),
                              total=int(np.ceil(combs/batch_size)), colour='green',
                              ncols=111, desc='Calculating UE p-values'):
                dead = Pool(processes=ncores)
                subtable = np.array(dead.map(self.ueFunc, batch))
                subtable = subtable[subtable[:, -1] < threshold]
                table.append(subtable)
            table = np.row_stack(table)
            print(f"UE probabilities ready [tte = {(time.time() - start) // 60} minutes]")
        df = pd.DataFrame(table, columns=['source', 'target', 'pvalue'])
        df['source'] = df['source'].replace(self.idx_to_proj)
        df['target'] = df['target'].replace(self.idx_to_proj)
        t = np.arange(1, df.shape[0] + 1) * threshold * 2 / (self.N * (self.N - 1))
        df = df.sort_values('pvalue')
        fdr_df = df[df['pvalue'] < t]
        fdr_df = fdr_df.assign(threshold=t[:fdr_df.shape[0]])
        if fdr_save_path is not None:
            fdr_df.to_csv(fdr_save_path, sep=sep, index=None)
        print(f"FDR applied! {fdr_df.shape[0]} links have been validated!")
        t = threshold * 2 / (self.N * (self.N - 1))
        bon_df = fdr_df[fdr_df['pvalue'] < t]
        bon_df = bon_df.assign(threshold=np.ones(bon_df.shape[0]) * t)
        if bon_save_path is not None:
            bon_df.to_csv(bon_save_path, sep=sep, index=None)
        print(f"Bonferroni applied! {bon_df.shape[0]} links have been validated!")
        return fdr_df, bon_df
    
    @staticmethod
    def batchIterator(iterator, batch_size):
    	# helper function to split the validation into batches for under-expressions
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
