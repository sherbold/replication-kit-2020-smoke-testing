package weka.classifiers.rules;

import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.rules.TestName;
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
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import org.apache.commons.math3.stat.inference.TestUtils;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.Instance;



/**
 * Automatically generated smoke and metamorphic tests.
 */
@Generated("atoml.testgen.TestclassGenerator")
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
@RunWith(Parameterized.class)
public class WEKA_RIPPER_AtomlTest {

    
    
    @Rule
    public TestName testname = new TestName();

    @Parameters(name = "{1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            { new String[]{"-E","-F","3","-N","2.0","-O","2"}, "-E -F 3 -N 2.0 -O 2"},
            { new String[]{"-F","3","-N","2.0","-O","2"}, "-F 3 -N 2.0 -O 2"},
            { new String[]{"-F","1","-N","2.0","-O","2"}, "-F 1 -N 2.0 -O 2"},
            { new String[]{"-F","2","-N","2.0","-O","2"}, "-F 2 -N 2.0 -O 2"},
            { new String[]{"-F","3","-N","1.0","-O","2"}, "-F 3 -N 1.0 -O 2"},
            { new String[]{"-F","3","-N","3.0","-O","2"}, "-F 3 -N 3.0 -O 2"},
            { new String[]{"-F","3","-N","2.0","-O","1"}, "-F 3 -N 2.0 -O 1"},
            { new String[]{"-F","3","-N","2.0","-O","3"}, "-F 3 -N 2.0 -O 3"},
            { new String[]{"-P","-F","3","-N","2.0","-O","2"}, "-P -F 3 -N 2.0 -O 2"}
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
    
    private Instances loadData(String resourceName) {
        Instances data;
        InputStreamReader originalFile = new InputStreamReader(
                 this.getClass().getResourceAsStream(resourceName));
        try(BufferedReader reader = new BufferedReader(originalFile);) {
            data = new Instances(reader);
            reader.close();
        }
        catch (IOException e) {
            throw new RuntimeException(resourceName, e);
        }
        data.setClassIndex(data.numAttributes()-1);
        return data;
    }

    @Test(timeout=21600000)
    public void test_Uniform() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Uniform_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Uniform_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Categorical() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Categorical_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Categorical_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_MinFloat() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/MinFloat_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/MinFloat_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_VerySmall() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/VerySmall_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/VerySmall_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_MinDouble() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/MinDouble_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/MinDouble_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_MaxFloat() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/MaxFloat_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/MaxFloat_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_VeryLarge() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/VeryLarge_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/VeryLarge_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_MaxDouble() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/MaxDouble_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/MaxDouble_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Split() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Split_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Split_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_LeftSkew() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/LeftSkew_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/LeftSkew_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_RightSkew() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/RightSkew_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/RightSkew_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_OneClass() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/OneClass_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/OneClass_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Bias() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Bias_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Bias_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Outlier() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Outlier_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Outlier_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Zeroes() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Zeroes_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Zeroes_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_RandomNumeric() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/RandomNumeric_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/RandomNumeric_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_RandomCategorial() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/RandomCategorial_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/RandomCategorial_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_DisjointNumeric() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/DisjointNumeric_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/DisjointNumeric_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_DisjointCategorical() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/DisjointCategorical_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/DisjointCategorical_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_StarvedMany() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/StarvedMany_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/StarvedMany_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_StarvedBinary() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/StarvedBinary_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/StarvedBinary_" + iter + "_test.arff");
            Classifier classifier = AbstractClassifier.forName("weka.classifiers.rules.JRip", Arrays.copyOf(parameters, parameters.length));
            classifier.buildClassifier(data);
            for (Instance instance : testdata) {
                classifier.classifyInstance(instance);
                classifier.distributionForInstance(instance);
            }
        }
    }


}