#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
from datetime import datetime

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory

"""
This class is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self,sentences, base_dir):
        try:
            lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        except:
            pass
        analyzer = StandardAnalyzer()
        storeDir = os.path.join(base_dir,INDEX_DIR)
        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        store = SimpleFSDirectory(Paths.get(storeDir))
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        self.indexsents(sentences, writer)
        # ticker = Ticker()
        # print 'commit index'
        # threading.Thread(target=ticker.run).start()
        # writer.commit()
        # writer.close()
        # ticker.tick = False
        # print 'done'

    def indexsents(self,sentences, writer):

        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        t2 = FieldType()
        t2.setStored(False)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        for i,sent in enumerate(sentences):
            #print "adding",i, sent
            try:
                root = os.getcwd()
                #contents = unicode(sent, 'iso-8859-1')
                doc = Document()
                doc.add(Field("name", str(i), t1))
                doc.add(Field("path", root, t1))
                if len(sent) > 0:
                    doc.add(Field("contents",sent.lower(), t2))
                else:
                    print "warning: no content in %s" % str(i)
                writer.addDocument(doc)
            except Exception, e:
                print "Failed in indexsents:", e
        writer.commit()
        writer.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print IndexFiles.__doc__
        sys.exit(1)

    print 'lucene', lucene.VERSION
    start = datetime.now()
    try:
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        IndexFiles(sys.argv[1],base_dir)
        end = datetime.now()
        print end - start
    except Exception, e:
        print "Failed: ", e
        raise e
