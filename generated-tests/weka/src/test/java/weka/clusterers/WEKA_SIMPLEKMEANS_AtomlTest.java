package weka.clusterers;

import static org.apache.commons.math3.stat.inference.TestUtils.tTest;
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
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;
import com.google.common.primitives.Doubles;
import smile.stat.hypothesis.KSTest;

import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;



/**
 * Automatically generated smoke and metamorphic tests.
 */
@Generated("atoml.testgen.TestclassGenerator")
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
@RunWith(Parameterized.class)
public class WEKA_SIMPLEKMEANS_AtomlTest {

    

    @Rule
    public TestName testname = new TestName();

    @Parameters(name = "{1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","1","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 1 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","2","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 2 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","3","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 3 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","200","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 200 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","100","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 100 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","550","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 550 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","0.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 0.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","4.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 4.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-V","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -V -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-M","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -M -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","3"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 3"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","4"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 4"},
            { new String[]{"-init","0","-A","weka.core.ManhattanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.ManhattanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","100","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 100 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","900","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 900 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2","-O"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2 -O"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-fast","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","1","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -fast -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 1 -I 500 -t1 -1.25 -t2 -1.0 -N 2"},
            { new String[]{"-init","0","-A","weka.core.EuclideanDistance -R first-last","-C","-periodic-pruning","1000","-min-density","2.0","-max-candidates","100","-num-slots","2","-I","500","-t1","-1.25","-t2","-1.0","-N","2"}, "-init 0 -A weka.core.EuclideanDistance -R first-last -C -periodic-pruning 1000 -min-density 2.0 -max-candidates 100 -num-slots 2 -I 500 -t1 -1.25 -t2 -1.0 -N 2"}
           });
    }

    @Parameter
    public String[] parameters;

    @Parameter(1)
    public String parameterName;

    private void assertMorphTest(String evaluationType, String testcaseName, int iteration, int testsize, int deviationsCounts, double[] deviationVector, int deviationsScores, HashMap<Integer, ArrayList<Double>> expectedScoresMap, HashMap<Integer, ArrayList<Double>> morphedScoresMap, Boolean passed, String errorMessage, String exception, String stacktrace) {
        if (passed) {
            if( "clust_exact".equalsIgnoreCase(evaluationType) ) {
                String message = String.format("clusters different (deviations of instances: %d out of %d)", deviationsCounts, testsize);
                assertTrue(message, deviationsCounts==0);
            }
            else if( "clust_stat".equalsIgnoreCase(evaluationType) ) {
                double pValueCounts;
                if( deviationsCounts>0 ) {
                    pValueCounts = tTest(0.0, deviationVector);
                } else {
                    pValueCounts = 1.0;
                }
                String message = String.format("results significantly different, p-value = %f (deviations of instances: %d out of %d)", pValueCounts, deviationsCounts, testsize);
                assertTrue(message, pValueCounts>0.05);
            }
            else if ("score_stat".equalsIgnoreCase(evaluationType)) {
                if (expectedScoresMap.isEmpty()) {
                    throw new RuntimeException("no scores found! score-matching available only for soft clustering algorithms!");
                }
                double[] pValuesKS = new double[expectedScoresMap.size()];
                int deviationsPValues = 0;
                for (Integer cluster: expectedScoresMap.keySet()) {
                    if (morphedScoresMap.containsKey(cluster) && morphedScoresMap.get(cluster).size() > 1 && expectedScoresMap.get(cluster).size() > 1) {
                        pValuesKS[cluster] = KSTest.test(Doubles.toArray(expectedScoresMap.get(cluster)), Doubles.toArray(morphedScoresMap.get(cluster))).pvalue;
                    } else {
                        pValuesKS[cluster] = 1.0;
                    }
                    if (pValuesKS[cluster] < (0.05 / (double) expectedScoresMap.size())) {
                        deviationsPValues++;
                    }
                }
                String message = String.format("scores significantly different (deviations of scores: %d out of %d, %d of %d clusters with significantly different scores)", deviationsScores, testsize, deviationsPValues, expectedScoresMap.size());
                assertTrue(message, deviationsScores==0);
            }
            else {
                throw new RuntimeException("invalid evaluation type for morph test: " + evaluationType + " (allowed: clust_exact, clust_stat, score_stat)");
            }
        } else {
            String message = errorMessage + '\n' + exception + '\n' + stacktrace;
            assertTrue(message, passed);
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

        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (data.classIndex() + 1));
        try {
            filter.setInputFormat(data);
            data = Filter.useFilter(data, filter);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        return data;
    }

    private HashMap<Integer, HashSet<Integer>> flipSameClusters(HashMap<Integer, HashSet<Integer>> expectedClustersMap, HashMap<Integer, HashSet<Integer>> morphedClustersMap) {
        HashMap<Integer, HashSet<Integer>> flippedMap = new HashMap<Integer, HashSet<Integer>>();
        for (int i = 0; i < morphedClustersMap.size(); i++) {
            boolean flipped = false;
            for (int j = 0; j < expectedClustersMap.size(); j++) {
                if (morphedClustersMap.get(i).equals(expectedClustersMap.get(j))) {
                    flippedMap.put(j, new HashSet<>(expectedClustersMap.get(j)));
                    flipped = true;
                    break;
                }
            }
            if (!flipped) {
                flippedMap.put(i, new HashSet<>(morphedClustersMap.get(i)));
            }
        }
        return flippedMap;
    }

    private HashMap<Integer, HashSet<Integer>> createClusterMap(int numClusters) {
        HashMap<Integer, HashSet<Integer>> clusterMap = new HashMap<Integer, HashSet<Integer>>();
        for (int i = 0; i < numClusters; i++) {
            HashSet<Integer> clusterIndices = new HashSet<Integer>();
            clusterMap.put(i, clusterIndices);
        }
        return clusterMap;
    }

    private HashMap<Integer, ArrayList<Double>> createScoresMap(HashMap<Integer, HashSet<Integer>> clusterMap, double[] scores) {
		HashMap<Integer, ArrayList<Double>> scoresMap = new HashMap<>();
		for (Integer cluster: clusterMap.keySet()) {
			ArrayList<Double> clusterScores = new ArrayList<>();
			for (Integer instance: clusterMap.get(cluster)) {
				clusterScores.add(scores[instance]);
			}
			scoresMap.put(cluster, clusterScores);
		}
		return scoresMap;
	}

    @Test(timeout=21600000)
    public void test_Uniform() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Uniform_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Uniform_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Categorical() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Categorical_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Categorical_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_MinFloat() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/MinFloat_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/MinFloat_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_VerySmall() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/VerySmall_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/VerySmall_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_MinDouble() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/MinDouble_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/MinDouble_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_MaxFloat() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/MaxFloat_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/MaxFloat_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_VeryLarge() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/VeryLarge_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/VeryLarge_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_MaxDouble() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/MaxDouble_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/MaxDouble_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Split() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Split_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Split_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_LeftSkew() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/LeftSkew_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/LeftSkew_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_RightSkew() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/RightSkew_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/RightSkew_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_OneClass() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/OneClass_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/OneClass_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Bias() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Bias_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Bias_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Outlier() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Outlier_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Outlier_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_Zeroes() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/Zeroes_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/Zeroes_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_RandomNumeric() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/RandomNumeric_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/RandomNumeric_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_RandomCategorial() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/RandomCategorial_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/RandomCategorial_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_DisjointNumeric() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/DisjointNumeric_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/DisjointNumeric_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_DisjointCategorical() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/DisjointCategorical_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/DisjointCategorical_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_StarvedMany() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/StarvedMany_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/StarvedMany_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }

    @Test(timeout=21600000)
    public void test_StarvedBinary() throws Exception {
        for(int iter=1; iter<=1; iter++) {
            Instances data = loadData("/smokedata/StarvedBinary_" + iter + "_training.arff");
            Instances testdata = loadData("/smokedata/StarvedBinary_" + iter + "_test.arff");

            Clusterer clusterer = AbstractClusterer.forName("weka.clusterers.SimpleKMeans", Arrays.copyOf(parameters, parameters.length));
            clusterer.buildClusterer(data);
            for (Instance instance : testdata) {
               clusterer.clusterInstance(instance);
                clusterer.distributionForInstance(instance);
            }
        }
    }


}