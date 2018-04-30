package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW3 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances trainingAutoPrice = loadData("/Users/yogev/Google Drive/IDC/Year 2/Semester 2/Machine Learning from Data/HW/3/HomeWork3/src/HomeWork3/auto_price.txt");
		FeatureScaler.scaleData(trainingAutoPrice);

//		for (Knn.WeightingScheme dir : WeightingScheme.values()) {
//			// do what you want
//		}

		Knn knn = new Knn();
		knn.buildClassifier(trainingAutoPrice);


		knn.setUp(Knn.WeightingScheme.Weighted, Knn.LpDistance.Infinity, Knn.DistanceCheck.Regular, 10);

		System.out.println(knn.crossValidationError(trainingAutoPrice, 10));


	}

}
