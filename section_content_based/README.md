## Content based identification
This directory is used to train and test section content based identification.

##### 1. Dataset construction
Construct dataset based on pre-defined section headers.

    python data_construction.py [pre-defined section headers][index_path][path to save dataset] > run.log 

The format of built dataset is a json file:
    
    {data:
        [
            {'header':header1, 'content':content1}
            {'header':header2, 'content':content2}
                        ... ... ...
            {'header':headerN, 'content':contentN}

        ]
    }




