package org.apache.spark.ml.classification;

import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.rules.TestName;
import org.junit.After;
import org.junit.Before;
import org.junit.FixMethodOrder;
import org.junit.Rule;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;



import javax.annotation.Generated;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.io.IOException;
import weka.core.Attribute;
import weka.core.Instances;
import org.apache.commons.math3.stat.inference.TestUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;



/**
 * Automatically generated smoke and metamorphic tests.
 */
@Generated("atoml.testgen.TestclassGenerator")
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
@RunWith(Parameterized.class)
public class SPARK_NaiveBayes_Gaussian_AtomlTest {

    
    
    @Rule
    public TestName testname = new TestName();

    @Parameters(name = "{1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            { new String[]{"setSmoothing","1.0","setModelType","gaussian"}, "setSmoothing 1.0 setModelType gaussian"},
            { new String[]{"setSmoothing","0.0","setModelType","gaussian"}, "setSmoothing 0.0 setModelType gaussian"},
            { new String[]{"setSmoothing","0.5","setModelType","gaussian"}, "setSmoothing 0.5 setModelType gaussian"}
           });
    }
    
    @Parameter
    public String[] parameters;
    
    @Parameter(1)
    public String parameterName;

    private void assertMorphTest(String evaluationType, String testcaseName, int iteration, int deviationsCounts, int deviationsScores, int testsize, long[] expectedMorphCounts, long[] morphedCounts, double[] expectedMorphedDistributions, double[] morphedDistributions) {
        if( "score_exact".equalsIgnoreCase(evaluationType) ) {
            String message = String.format("results different (deviations of scores: %d out of %d)", deviationsScores, testsize);
            assertTrue(message, deviationsScores==0);
        }
        else if( "class_exact".equalsIgnoreCase(evaluationType) ) {
            String message = String.format("results different (deviations of classes: %d out of %d)", deviationsCounts, testsize);
            assertTrue(message, deviationsCounts==0);
        }
        else if( "class_stat".equalsIgnoreCase(evaluationType) ) {
            double pValueCounts;
            if( deviationsCounts>0 ) {
                pValueCounts = TestUtils.chiSquareTestDataSetsComparison(expectedMorphCounts, morphedCounts);
            } else {
                pValueCounts = 1.0;
            }
            String message = String.format("results significantly different, p-value = %f (deviations of classes: %d out of %d)", pValueCounts, deviationsCounts, testsize);
            assertTrue(message, pValueCounts>0.05);
        } 
        else if( "score_stat".equalsIgnoreCase(evaluationType) ) {
            double pValueKS;
            if( deviationsScores>0 ) {
                pValueKS = TestUtils.kolmogorovSmirnovTest(expectedMorphedDistributions, morphedDistributions);
            } else {
                pValueKS = 1.0;
            }
            String message = String.format("score distributions significantly different, p-value = %f (deviations of scores: %d out of %d)", pValueKS, deviationsScores, testsize);
            assertTrue(message, pValueKS>0.05);
        } else {
            throw new RuntimeException("invalid evaluation type for morph test: " + evaluationType + " (allowed: exact, classification, score)");
        }
    }
    

    private SparkSession sparkSession;
    
    @Before
    public void setUp() {
        sparkSession = SparkSession.builder().appName("Logistic_Default_AtomlTest").master("local[1]").getOrCreate();
        sparkSession.sparkContext().setLogLevel("WARN");
    }
    
    @After
    public void tearDown() {
        sparkSession.stop();
    }

    private Dataset<Row> arffToDataset(String filename) {
        Instances data;
        InputStreamReader file = new InputStreamReader(this.getClass().getResourceAsStream(filename));
        try (BufferedReader reader = new BufferedReader(file);) {
            data = new Instances(reader);
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException(filename, e);
        }

        List<StructField> fields = new LinkedList<>();
        for (int j = 0; j < data.numAttributes(); j++) {
            fields.add(DataTypes.createStructField(getNormalizedName(data.attribute(j)), DataTypes.DoubleType, false));
        }
        StructType schema = DataTypes.createStructType(fields);
        List<Row> rows = new LinkedList<>();
        for (int i = 0; i < data.size(); i++) {
            List<Double> valueList = new ArrayList<>(data.numAttributes());
            for (int j = 0; j < data.numAttributes(); j++) {
                valueList.add(data.instance(i).value(j));
            }

            rows.add(RowFactory.create(valueList.toArray()));
        }
        Dataset<Row> dataframe = sparkSession.createDataFrame(rows, schema);

        List<String> featureNames = new ArrayList<>();
        List<String> nominals = new LinkedList<>();
        List<String> nominalsOutput = new LinkedList<>();
        for (int j = 0; j < data.numAttributes() - 1; j++) {
            featureNames.add(getNormalizedName(data.attribute(j)));
            if (data.attribute(j).isNominal()) {
                nominals.add(getNormalizedName(data.attribute(j)));
                nominalsOutput.add(getNormalizedName(data.attribute(j)) + "_onehot");
            }
        }
        if (!nominals.isEmpty()) {
            OneHotEncoder oneHot = new OneHotEncoder().setInputCols(nominals.toArray(new String[0]))
                    .setOutputCols(nominalsOutput.toArray(new String[0])).setDropLast(false);
            OneHotEncoderModel oneHotModel = oneHot.fit(dataframe);
            dataframe = oneHotModel.transform(dataframe);
            dataframe = dataframe.drop(nominals.toArray(new String[0]));
            for (int j = nominals.size() - 1; j >= 0; j--) {
                dataframe = dataframe.withColumnRenamed(nominalsOutput.get(j), nominals.get(j));
            }
        }
        VectorAssembler va = new VectorAssembler().setInputCols(featureNames.toArray(new String[0]))
                .setOutputCol("features");
        dataframe = va.transform(dataframe);

        return dataframe;
    }
    
    private String getNormalizedName(Attribute attribute) {
        return attribute.name().replaceAll("\\.", "_");
    }
    
    private void setParameters(Object classifier, String[] parameters) {
        Method[] methods = classifier.getClass().getMethods();
        for( int i=0; i<parameters.length; i=i+2) {
            boolean methodFound = false;
            for(Method method : methods) {
                if( method.getName().equals(parameters[i])) {
                    methodFound = true;
                    for( java.lang.reflect.Parameter param : method.getParameters()) {
                        try {
                            if( "long".equals(param.getType().getName()) ) {
                                method.invoke(classifier, Long.parseLong(parameters[i+1]));
                                
                            }
                            else if( "int".equals(param.getType().getName()) ) {
                                method.invoke(classifier, Integer.parseInt(parameters[i+1]));
                            }
                            else if( "double".equals(param.getType().getName()) ) {
                                method.invoke(classifier, Double.parseDouble(parameters[i+1]));
                            }
                            else if( "boolean".equals(param.getType().getName()) ) {
                                method.invoke(classifier, Boolean.parseBoolean(parameters[i+1]));
                            }
                            else if( "java.lang.String".equals(param.getType().getName()) ) {
                                method.invoke(classifier, parameters[i+1]);
                            }
                            else if( "[I".equals(param.getType().getName()) ) {
                                String[] strNumbers = parameters[i+1].split(",");
                                int[] intArray = new int[strNumbers.length];
                                for( int j=0; j<strNumbers.length; j++) {
                                    intArray[j] = Integer.parseInt(strNumbers[j].trim());
                                }
                                method.invoke(classifier, intArray);
                            }
                            else {
                                throw new RuntimeException("Hyperparameter type not supported: " + param.getType().getName());
                            }
                        } catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
                            throw new RuntimeException("Failure instantiating hyperparameter: " + parameters[i]);
                        }
                    }
                }
            }
            if( !methodFound ) {
                throw new RuntimeException("Invalid hyperparameters generated by atoml");
            }
        }
    }

    @Test
    public void test_Uniform() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Uniform_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Uniform_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Categorical() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Categorical_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Categorical_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_MinFloat() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/MinFloat_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/MinFloat_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_VerySmall() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/VerySmall_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/VerySmall_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_MinDouble() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/MinDouble_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/MinDouble_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_MaxFloat() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/MaxFloat_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/MaxFloat_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_VeryLarge() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/VeryLarge_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/VeryLarge_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_MaxDouble() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/MaxDouble_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/MaxDouble_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Split() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Split_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Split_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_LeftSkew() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/LeftSkew_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/LeftSkew_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_RightSkew() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/RightSkew_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/RightSkew_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_OneClass() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/OneClass_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/OneClass_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Bias() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Bias_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Bias_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Outlier() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Outlier_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Outlier_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Zeroes() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Zeroes_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Zeroes_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_RandomNumeric() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/RandomNumeric_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/RandomNumeric_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_RandomCategorial() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/RandomCategorial_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/RandomCategorial_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_DisjointNumeric() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/DisjointNumeric_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/DisjointNumeric_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_DisjointCategorical() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/DisjointCategorical_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/DisjointCategorical_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_StarvedMany() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/StarvedMany_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/StarvedMany_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_StarvedBinary() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/StarvedBinary_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/StarvedBinary_" + iter + "_test.arff");
            
            NaiveBayes classifier = new NaiveBayes();
            classifier.setLabelCol("classAtt");
            try {
            	Method setSeedMethod = classifier.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(classifier, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(classifier, parameters);
            
            ClassificationModel<?, ?> model = classifier.fit(dataframe);
            model.transform(testdata);
        }
    }


}