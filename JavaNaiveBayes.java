package CS286;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;


import scala.Tuple2;

public class JavaNaiveBayes implements Serializable{

    static HashMap<String,Double> labeledData;
    static List<String> hamdata;
    static List <String> spamdata;

    public static void main(String[] args) {
        labeledData =new  HashMap<String,Double>();
        hamdata =new ArrayList();
        spamdata =new ArrayList();
        for(String arg: args){

            System.out.println(arg);

        }

        JavaNaiveBayes jnb =new JavaNaiveBayes();
        SparkConf sparkConf = new SparkConf().setAppName("JavaBookExample").setMaster("local[1]");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        if(args[0].equals("build"))
        {
            jnb.cleanData(args[1],args[2]);
            jnb.buildModel(args[3], sc);
        }
        else if(args[0].equals("predict"))
        {
            jnb.predictValue(args[1], sc, jnb.getSubjectline(args[2]));
        }

        sc.stop();
    }

    public String getSubjectline(String path)
    {
        String line="";
        File file =new File(path);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            while ((line = br.readLine()) != null) {
                if (line.split(" ")[0].toLowerCase().contains("subject")) {
                   return line.replace("Subject: ", "").replaceAll("[^a-zA-Z0-9 ]+","");
                }
            }
        } catch (Exception e) {

        }
        return null;
    }

    public void cleanData(String hamPath,String spamPath)
    {

        listAllSubject(spamPath,0.0);
        listAllSubject(hamPath,1.0);
        WriteToFile();
       
    }

    public void listAllSubject(String path,double label) {
        File folder = new File(path);
        File[] listOfFiles = folder.listFiles();
        System.out.println("count of files:-->" + listOfFiles.length);
        for (File file : listOfFiles) {
            if (file.isFile()) {
                readFile(file,label);
            }
        }
    }

    public void WriteToFile()
    {
        String ham ="ham.txt";
        String spam ="spam.txt";
        File fileHam =new File(ham);
        if(fileHam.exists())
            fileHam.delete();
        FileWriter hamwriter;
        File fileSpam =new File(spam);
        if(fileSpam.exists())
            fileSpam.delete();
        FileWriter spamwriter;

        try {
            hamwriter = new FileWriter(ham, true);
            spamwriter =new FileWriter(spam,true);
            for (Map.Entry<String, Double> map : labeledData.entrySet()) {
                if(map.getValue()==1.0)
                {
                    hamwriter.write(map.getKey());
                    hamwriter.write("\n");
                    hamdata.add(map.getKey());
                }
                else
                {
                    spamwriter.write(map.getKey());
                    spamwriter.write("\n");
                    spamdata.add(map.getKey());
                    
                }


            }
            hamwriter.close();
            spamwriter.close();
        } catch (Exception e) {

        }
    }

    public void readFile(File file,double label) {
        String line="";
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            while ((line = br.readLine()) != null) {
                if (line.split(" ")[0].toLowerCase().contains("subject")) {
                    labeledData.put(line.replace("Subject: ", "").replaceAll("[^a-zA-Z0-9 ]+",""),label);
                }
            }
        } catch (Exception e) {

        }
    }

    public void buildModel(String modelPath,JavaSparkContext sc)
    {


        // Load 2 types of emails from text files: spam and ham (non-spam).
        // Each line has text from one email.
        JavaRDD<String> spam = sc.parallelize(spamdata) ;
        JavaRDD<String> ham = sc.parallelize(hamdata);

        // Create a HashingTF instance to map email text to vectors of 100 features.
        final HashingTF tf = new HashingTF(100);

        // Each email is split into words, and each word is mapped to one feature.
        // Create LabeledPoint datasets for positive (spam) and negative (ham) examples.
        JavaRDD<LabeledPoint> positiveExamples = spam.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String email) {
                //]]System.out.println(tf.transform(Arrays.asList(email.split(" "))));
                Vector dataMatrix =tf.transform(Arrays.asList(email)).toDense();

                return new LabeledPoint(1, dataMatrix);
            }
        });
        JavaRDD<LabeledPoint> negativeExamples = ham.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String email) {
                Vector dataMatrix =tf.transform(Arrays.asList(email)).toDense();
                return new LabeledPoint(0, dataMatrix);
            }
        });
        JavaRDD<LabeledPoint> totalData = positiveExamples.union(negativeExamples);
        // trainingData.cache(); // Cache data since Logistic Regression is an iterative algorithm.
        



        JavaRDD<LabeledPoint>[] tmp = totalData.randomSplit(new double[]{0.6, 0.4});
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set

        final NaiveBayesModel model = NaiveBayes.train (training.rdd(),1.0);
        JavaPairRDD<Double, Double> predictionAndLabel =
                test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });
        double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Double, Double> pl) {
                return pl._1().equals(pl._2());
            }
        }).count() / (double) test.count();
        System.out.println("accuracy="+accuracy*100+"%");
        // Test on a positive example (spam) and a negative one (ham).
        // First apply the same HashingTF feature transformation used on the training data.

        String s ="Attn: How about being a few pounds lighter?OLIBYKN";
        model.save(sc.sc(), modelPath);
    }

    public void predictValue(String modelPath,JavaSparkContext sc,String s)
    {
        final HashingTF tf = new HashingTF(100);
        NaiveBayesModel sameModel = NaiveBayesModel.load(sc.sc(), modelPath);
        Vector posTestExample =
                tf.transform(Arrays.asList(s.split(" "))).toDense();
        // Now use the learned model to predict spam/ham for new emails.
        System.out.println("spam 1 ham 0: "+s+"   -->" + sameModel.predict(posTestExample));
        if(sameModel.predict(posTestExample)==1)
        {
            System.out.println("Classify : Spam");
        }
        else
        {
            System.out.println("Classify : Ham");
        }

    }
}
