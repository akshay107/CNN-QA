import sys, os, lucene

from string import Template
from datetime import datetime
from getopt import getopt, GetoptError

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import SimpleFSDirectory

class retrieve_sents():

    def __init__(self,indexDir,query):
        self.indexDir = indexDir
        self.query = query
        # lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        self.stats = False

    def retrieve_sents(self):

        indexDir = self.indexDir
        query = self.query

        sent_ind_list = []
        # template = CustomTemplate(format)
        fsDir = SimpleFSDirectory(Paths.get(indexDir))
        # print indexDir
        searcher = IndexSearcher(DirectoryReader.open(fsDir))

        analyzer = StandardAnalyzer()
        parser = QueryParser("contents", analyzer)
        parser.setDefaultOperator(QueryParser.Operator.OR)
        query = parser.parse(query)
        # print query
        start = datetime.now()
        scoreDocs = searcher.search(query, 50).scoreDocs
        duration = datetime.now() - start
        # print query
        if self.stats:
            print >>sys.stderr, "Found %d sentences (in %s) that matched query '%s':" %(len(scoreDocs), duration, query)

        for scoreDoc in scoreDocs:
            # print scoreDoc.doc
            # doc = searcher.doc(scoreDoc.doc)
            sent_ind_list.append(scoreDoc.doc)

        return sent_ind_list


if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
