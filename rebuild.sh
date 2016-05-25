#!/bin/bash

mkdir -p src/main/java/CS286

cp -f MyNaiveBayes.java src/main/java/CS286
cp -f JavaNaiveBayes.java src/main/java/CS286/

mvn clean 
mvn install

cp -f target/NaiveBayes.jar /root/
cp -f target/NaiveBayes.jar .
