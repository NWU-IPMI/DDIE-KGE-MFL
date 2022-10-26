# -*- coding: utf-8 -*-
import stanza

def download_stanza(path):
    stanza.download('en',package='craft',model_dir=path)

if __name__=="__main__":
    download_stanza("./stanza_resources/")

