package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class DistanceCalculator {



    /**
    * We leave it up to you wheter you want the distance method to get all relevant
    * parameters(lp, efficient, etc..) or have it has a class variables.
    */
    public double distance (Instance one, Instance two, int p) {
        return lpDistance(one, two, p);
    }

    public double distance (Instance one, Instance two, double threshold, int p) {
        return efficientLpDistance(one, two, threshold, p);
    }

    public double distance (Instance one, Instance two) {
        return lInfinityDistance(one, two);
    }

    public double distance (Instance one, Instance two, double threshold) {
        return efficientLInfinityDistance(one, two, threshold);
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two, int p) {
        int numOfAttributes = one.numAttributes() - 1;
        double absoluteValue, powerOfDifference, sum = 0;

        for (int i = 0; i < numOfAttributes; i++) {

            powerOfDifference = Math.pow((one.value(i) - two.value(i)), p);
            absoluteValue = Math.abs(powerOfDifference);
            sum += absoluteValue;
        }
        return Math.pow(sum, (1 / p));
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        int numOfAttributes = one.numAttributes() - 1;
        double max = 0, different;

        for (int i = 0; i < numOfAttributes; i++) {
            different = Math.abs(one.value(i) - two.value(i));

            if (different > max){
                max = different;
            }
        }
        return max;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDistance(Instance one, Instance two, double threshold, int p) {
        int numOfAttributes = one.numAttributes() - 1;
        double absoluteValue, powerOfDifference, sum = 0;


        for (int i = 0; i < numOfAttributes; i++) {

            powerOfDifference = Math.pow((one.value(i) - two.value(i)), p);
            absoluteValue = Math.abs(powerOfDifference);
            sum += absoluteValue;

            if (sum > threshold){
                sum = Double.MAX_VALUE;
                break;
            }
        }

        if (sum != Double.MAX_VALUE){
            return sum;
        }
        else{
            return Math.pow(sum, (1 / p));
        }
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two, double threshold) {
        int numOfAttributes = one.numAttributes() - 1;
        double max = 0, different;

        for (int i = 0; i < numOfAttributes; i++) {
            different = Math.abs(one.value(i) - two.value(i));

            if (different > max){
                max = different;

                if (max > threshold){
                    max = Double.MAX_VALUE;
                }
            }

        }
        return max;
    }
}

public class Knn implements Classifier {

    public enum DistanceCheck{Regular, Efficient}
    private Instances m_trainingInstances;

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {

    }

    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        return 0.0;
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @param insatnces
     * @return
     */
    public double calcAvgError (Instances insatnces){
        return 0.0;
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param insances Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances insances, int num_of_folds){
        return 0.0;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public void findNearestNeighbors(Instance instance) {
        // TODO: 29/04/2018  Return type: /* Collection of your choice */
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (/* Collection of your choice */) {
        return 0.0;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(/* Collection of your choice */) {
        return 0.0;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}
