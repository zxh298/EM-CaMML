package camml.core.library;


import cdms.core.Value;

public class SelectedVectorLatent extends SelectedVector {

	/** copy the data weight from the original vector */
	protected double[] originalWeight;

	public SelectedVectorLatent(Vector v) {
		super(v);
		// copy the weight from the original vector (data)
		originalWeight = new double[v.length()];
		for (int i = 0; i < v.length(); i++) {
			originalWeight[i] = v.weight(i);
		}
	}

	public SelectedVectorLatent(Value.Vector v, final int[] row, final int[] column) {
		super(v, row, column);
		// copy the weight from the original vector (data)
		originalWeight = new double[v.length()];
		for (int i = 0; i < v.length(); i++) {
			originalWeight[i] = v.weight(i);
		}
	}

	public Value.Vector getOriginalVector() {
		return originalVector;
	}

	public double getOriginalWeight(int i) {
		return originalWeight[i];
	}

}
