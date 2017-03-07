##RASIA
Research Articles' Structure Identification and Applications in academic text ming, bibliometrics and scientometrics. 

There are two main task in this project.

    1. Identification of General Structure of scientific articles.
    2. Applications in bibliometrics, scientometrics and text mining.

##Data

    1. Research articles from Computer Science, labeled as CS
    2. Articles from PLOS. PLOS ONE used, labeled as PLOS


##Method

    1. section header based identification 
    2. section content based identification 
    3. hybrid identification 

##Tools

1. [Scikit-learn](http://scikit-learn.org/stable/)  
2. [FastText](https://github.com/facebookresearch/fastText)  

##Directory
    |--statistics
        |--plos_xml_statistics.py: do some statistics of PLOS_XML data.
        |--sc_xml_statistics.py
    |--tools
        |--random_selection.py: random select N lines from given file.

##Usage

#####Preprocessing
data will be saved to data/sec-header.json and data/sec-type.json. The log info will print through standard outstream and data will be outputed through error stream. 

    python statistics/plos_xml_statistics.py [path direcotry] 1>plos_statstic.log 2>header_style.txt 

For scienceDirect data:

    python statistics/sc_xml_statistics.py [index file path] 1>headers.txt 2> sc_statistic.log 







