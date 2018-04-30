package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.Filter;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public static Instances scaleData(Instances instances) throws Exception {

		// TODO: 29/04/2018 check about the Exception, Static??
		Standardize filter = new Standardize();
		Instances defaultStdData;

		filter.setInputFormat(instances);
		defaultStdData = Filter.useFilter(instances, filter);

		return defaultStdData;

	}
}