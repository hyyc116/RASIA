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

    1. Scikit-learn 
    2. FastText 
    3. CRF 

##Directory
    |--tools
        |--plos_xml_statistics.py: do some statistics of PLOS_XML data.
        |--sc_xml_statistics.py

##Usage

#####Preprocessing
data will be saved to data/sec-header.json and data/sec-type.json. The log info will print through standard outstream and data will be outputed through error stream. 

    python tools/plos_xml_statistics.py [path direcotry] 1>plos_statstic.log 2>header_style.txt 







