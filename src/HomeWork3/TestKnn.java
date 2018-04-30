package HomeWork3;

import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.PriorityQueue;

public class TestKnn {
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

        Instances validationSet = trainingAutoPrice.testCV(10, 1);
        Instances trainingSet = trainingAutoPrice.trainCV(10, 1);

        Knn knn = new Knn();
        knn.buildClassifier(trainingSet);
        knn.setUp(Knn.WeightingScheme.Uniform, Knn.LpDistance.One, Knn.DistanceCheck.Regular, 10);




        double error1, pred;
//        PriorityQueue<Entry> que = knn.findNearestNeighbors(validationSet.get(1));
//        pred = knn.getAverageValue(que);

        error1 = knn.calcAvgError(validationSet);

        System.out.println(error1);


        Knn.WeightingScheme chosenWeightingScheme = null;
        Knn.LpDistance chosenP = null;
        int chosenK = 0;
        double bestError = Double.MAX_VALUE, error;



    }
}
