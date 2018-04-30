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
		Instances scaled_trainingAutoPrice = FeatureScaler.scaleData(trainingAutoPrice);

		Knn knn = new Knn();
		Knn scaled_knn = new Knn();

		knn.buildClassifier(trainingAutoPrice);
		scaled_knn.buildClassifier(scaled_trainingAutoPrice);

		Knn.WeightingScheme chosenWeightingScheme = null;
		Knn.LpDistance chosenP = null;
		int chosenK = 0;
		double bestError = Double.MAX_VALUE, error;

		Knn.WeightingScheme scaled_chosenWeightingScheme;
		Knn.LpDistance scaled_chosenP;
		int scaled_chosenK;
		double scaled_bestError = Double.MAX_VALUE;

		for (Knn.WeightingScheme weightingScheme : Knn.WeightingScheme.values()) {
			for (Knn.LpDistance p : Knn.LpDistance.values()) {
				for (int k = 1; k <= 20; k++) {
					knn.setUp(weightingScheme, p, Knn.DistanceCheck.Regular, k);
					error = knn.crossValidationError(trainingAutoPrice, 10);

					if (error < bestError){
						chosenWeightingScheme = weightingScheme;
						chosenK = k;
						chosenP = p;
						bestError = error;
					}
				}
			}
		}

		System.out.println("K = " + chosenK +", lp = "+ chosenP +
				", majority function = " + chosenWeightingScheme + ", Error = " + bestError);

//
//		knn.setUp(Knn.WeightingScheme.Weighted, Knn.LpDistance.Infinity, Knn.DistanceCheck.Regular, 10);
//		System.out.println(knn.crossValidationError(trainingAutoPrice, 10));
//
//		knn.setUp(Knn.WeightingScheme.Weighted, Knn.LpDistance.Infinity, Knn.DistanceCheck.Efficient, 10);
//		System.out.println(knn.crossValidationError(trainingAutoPrice, 10));










	}

}
