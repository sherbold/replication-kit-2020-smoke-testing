package org.apache.spark.ml.clustering;

import static org.apache.commons.math3.stat.inference.TestUtils.tTest;
import static org.junit.Assert.*;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.FixMethodOrder;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

import javax.annotation.Generated;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.LinkedList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Collection;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
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
import com.google.common.primitives.Doubles;
import smile.stat.hypothesis.KSTest;

import java.io.PrintWriter;
import java.io.StringWriter;



/**
 * Automatically generated smoke and metamorphic tests.
 */
@Generated("atoml.testgen.TestclassGenerator")
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
@RunWith(Parameterized.class)
public class SPARK_KMEANS_AtomlTest {

    

    @Rule
    public TestName testname = new TestName();

    @Parameters(name = "{1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            { new String[]{"tol","0.5","maxIter","50","k","2","distanceMeasure","euclidean","initSteps","2","initMode","k-means||"}, "tol 0.5 maxIter 50 k 2 distanceMeasure euclidean initSteps 2 initMode k-means||"},
            { new String[]{"tol","0.5","maxIter","50","k","3","distanceMeasure","euclidean","initSteps","2","initMode","k-means||"}, "tol 0.5 maxIter 50 k 3 distanceMeasure euclidean initSteps 2 initMode k-means||"},
            { new String[]{"tol","0.5","maxIter","50","k","4","distanceMeasure","euclidean","initSteps","2","initMode","k-means||"}, "tol 0.5 maxIter 50 k 4 distanceMeasure euclidean initSteps 2 initMode k-means||"},
            { new String[]{"tol","0.5","maxIter","0","k","2","distanceMeasure","euclidean","initSteps","2","initMode","k-means||"}, "tol 0.5 maxIter 0 k 2 distanceMeasure euclidean initSteps 2 initMode k-means||"},
            { new String[]{"tol","0.5","maxIter","100","k","2","distanceMeasure","euclidean","initSteps","2","initMode","k-means||"}, "tol 0.5 maxIter 100 k 2 distanceMeasure euclidean initSteps 2 initMode k-means||"},
            { new String[]{"tol","0.5","maxIter","50","k","2","distanceMeasure","euclidean","initSteps","2","initMode","random"}, "tol 0.5 maxIter 50 k 2 distanceMeasure euclidean initSteps 2 initMode random"},
            { new String[]{"tol","0.5","maxIter","50","k","2","distanceMeasure","euclidean","initSteps","1","initMode","k-means||"}, "tol 0.5 maxIter 50 k 2 distanceMeasure euclidean initSteps 1 initMode k-means||"},
            { new String[]{"tol","0.5","maxIter","50","k","2","distanceMeasure","euclidean","initSteps","3","initMode","k-means||"}, "tol 0.5 maxIter 50 k 2 distanceMeasure euclidean initSteps 3 initMode k-means||"},
            { new String[]{"tol","0.0","maxIter","50","k","2","distanceMeasure","euclidean","initSteps","2","initMode","k-means||"}, "tol 0.0 maxIter 50 k 2 distanceMeasure euclidean initSteps 2 initMode k-means||"},
            { new String[]{"tol","1.0","maxIter","50","k","2","distanceMeasure","euclidean","initSteps","2","initMode","k-means||"}, "tol 1.0 maxIter 50 k 2 distanceMeasure euclidean initSteps 2 initMode k-means||"},
            { new String[]{"tol","0.5","maxIter","50","k","2","distanceMeasure","cosine","initSteps","2","initMode","k-means||"}, "tol 0.5 maxIter 50 k 2 distanceMeasure cosine initSteps 2 initMode k-means||"}
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


    private static SparkSession sparkSession;

    @BeforeClass
    public static void setUpClass() {
        sparkSession = SparkSession.builder().appName("Logistic_Default_AtomlTest").master("local[1]").getOrCreate();
        sparkSession.sparkContext().setLogLevel("WARN");
    }

    @AfterClass
    public static void tearDownClass() {
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

    private void setParameters(Object clusterer, String[] parameters) {
    	Method[] methods = clusterer.getClass().getMethods();
        for( int i=0; i<parameters.length; i=i+2) {
        	boolean methodFound = false;
        	for(Method method : methods) {
        		if( method.getName().equals(parameters[i])) {
        			methodFound = true;
        			for( java.lang.reflect.Parameter param : method.getParameters()) {
        				try {
		    				if( "long".equals(param.getType().getName()) ) {
								method.invoke(clusterer, Long.parseLong(parameters[i+1]));

		    				}
		    				else if( "int".equals(param.getType().getName()) ) {
		    					method.invoke(clusterer, Integer.parseInt(parameters[i+1]));
		    				}
		    				else if( "double".equals(param.getType().getName()) ) {
		    					method.invoke(clusterer, Double.parseDouble(parameters[i+1]));
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

    private HashMap<Integer, ArrayList<Double>> createScoresMap(HashMap<Integer, HashSet<Integer>> clusterMap, List<Row> scores) {
		HashMap<Integer, ArrayList<Double>> scoresMap = new HashMap<>();
		for (Integer cluster: clusterMap.keySet()) {
			ArrayList<Double> clusterScores = new ArrayList<>();
			for (Integer instance: clusterMap.get(cluster)) {
				clusterScores.add(((DenseVector) scores.get(instance).get(0)).values()[cluster]);
			}
			scoresMap.put(cluster, clusterScores);
		}
		return scoresMap;
	}

    @Test
    public void test_Uniform() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Uniform_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Uniform_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_MinFloat() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/MinFloat_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/MinFloat_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_VerySmall() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/VerySmall_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/VerySmall_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_MinDouble() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/MinDouble_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/MinDouble_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_MaxFloat() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/MaxFloat_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/MaxFloat_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_VeryLarge() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/VeryLarge_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/VeryLarge_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_MaxDouble() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/MaxDouble_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/MaxDouble_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Split() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Split_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Split_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_LeftSkew() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/LeftSkew_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/LeftSkew_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_RightSkew() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/RightSkew_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/RightSkew_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_OneClass() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/OneClass_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/OneClass_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Bias() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Bias_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Bias_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Outlier() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Outlier_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Outlier_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_Zeroes() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/Zeroes_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/Zeroes_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_RandomNumeric() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/RandomNumeric_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/RandomNumeric_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }

    @Test
    public void test_DisjointNumeric() throws Exception {
        for(int iter=1; iter<=1; iter++) {
        	Dataset<Row> dataframe = arffToDataset("/smokedata/DisjointNumeric_" + iter + "_training.arff");
            Dataset<Row> testdata = arffToDataset("/smokedata/DisjointNumeric_" + iter + "_test.arff");

            KMeans clusterer = new KMeans();
            try {
            	Method setSeedMethod = clusterer.getClass().getMethod("setSeed", long.class);
            	setSeedMethod.invoke(clusterer, 42);
            } catch (NoSuchMethodException | SecurityException e) {
            	// not randomized
            }
            setParameters(clusterer, parameters);

            KMeansModel model = clusterer.fit(dataframe);
            model.transform(testdata);
        }
    }


}