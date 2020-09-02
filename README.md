# Introduction

Within this archive you find the replication package for the paper "Smoke Testing for Machine Learning: Simple Tests to Discover Severe Defects" by Steffen Herbold and Tobias Haar which is currently under review. The aim of this replication package is to allow other researchers to replicate our results with minimal effort. 

# Requirements
- Gradle
- Java 11
- Python 3.6

# Contents
This archive contains:
- The directory generated-tests with the JUnit/unittest test suites that were generated for Weka, scikit-learn and Spark MLlib. 
- The directory descriptions with the YAML descriptions of the algorithm that we use to automatically generate the tests. 
- A copy of atoml, the tool we used to automatically generate the tests. 

# How does it work
To replicate the test generation, you require the tool [atoml](https://github.com/sherbold/atoml), a copy is provided as Zip archive. The Project contains a gradle file that contains all required Java dependencies, including for the execution of the JUnit tests for Weka and Spark MLlib. You can build atoml in a Linux bash and generate the tests with the following commands.

```
unzip atoml.zip
cd atoml-master
./gradlew build
unzip build/distributions/atoml.zip -d ../atoml-build
cd ..
./atoml-build/atoml/bin/atoml -f descriptions/descriptions.yml -nomorph
```

The tests generated tests can be executed using JUnit, resp unittest. For The JUnit test, make sure that Weka (and the plug-ins for classifiers), resp. Spark and Spark ML are part of the classpath for the test execution. For scikit-learn, make sure that the latest version of scikit-learn and numpy are available. 

# License

This replication package is used are licensed under the Apache License, Version 2.0.
