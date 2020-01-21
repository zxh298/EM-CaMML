package camml.core.latentDetect;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

import camml.plugin.tetrad4.Tetrad4;
import camml.plugin.tetrad4.Tetrad4FN;
import camml.plugin.weka.Converter;
import cdms.core.Type;
import cdms.core.Value;
import cdms.core.VectorFN;
import edu.cmu.tetrad.data.RectangularDataSet;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.IndTestChiSquare;
import edu.cmu.tetrad.search.IndTestGSquare;
import edu.cmu.tetrad.search.IndependenceTest;

public class DataPreprocessing {

	// Data:
	Value.Vector data = null;
	// Get the data size (row number):
	private int dataSize;
	// Get the number of variables (only observed variables):
	private int variableNum;
	// Get the labels of all variables (only observed variables);
	private List<String> variableLabels;
	// Get the states of each variable (only observed variables):
	private ArrayList<ArrayList<String>> statesOfAllVariables;
	// Get the states number of every variable (only observed variables):
	private int[] stateNumOfAllVariables;
	// Get data of each variable (only observed variables):
	private ArrayList<ArrayList<String>> valuesOfAllVariables;
	// Dependency matrices of current data:
	private ArrayList<int[][]> currentDataDependencyMatrices;
	// Indicate the current data matches any trigger or not:
	private boolean isMatched;
	// To store the candidate hidden model of current data if
	// it matches any trigger:
	private int[][] candidateHiddenModel;
	// Get the indexes of the two children in the hidden common cause:
	private int[] latentVarChildrenIndexes;
	// Get all instances of the given data (only observed variables):
	private ArrayList<String[]> allInstances;
	// Get the number of different joint observations in the given data
	// (only observed variables and for calculating MDL score):
	private int jointObservationNum;

	private double errorRate;

	private double alpha;
	
	// whether to check of matching trigger
	private boolean runTriggerMatching;
	
	public DataPreprocessing(double errorRate, double alpha, boolean runTriggerMatching) {
		this.data = null;
		this.dataSize = 0;
		this.variableNum = 0;
		this.variableLabels = new ArrayList<String>();
		this.statesOfAllVariables = new ArrayList<ArrayList<String>>();
		this.valuesOfAllVariables = new ArrayList<ArrayList<String>>();
		this.currentDataDependencyMatrices = new ArrayList<int[][]>();
		this.allInstances = new ArrayList<String[]>();
		this.jointObservationNum = 0;
        this.errorRate = errorRate;
        this.alpha = alpha;
        this.runTriggerMatching = runTriggerMatching;
	}

	public DataPreprocessing(Value.Vector data, double errorRate, double alpha, boolean runTriggerMatching) {
		this.data = data;
		this.dataSize = 0;
		this.variableNum = 0;
		this.variableLabels = new ArrayList<String>();
		this.statesOfAllVariables = new ArrayList<ArrayList<String>>();
		this.valuesOfAllVariables = new ArrayList<ArrayList<String>>();
		this.currentDataDependencyMatrices = new ArrayList<int[][]>();
		this.allInstances = new ArrayList<String[]>();
		this.jointObservationNum = 0;
		this.errorRate = errorRate;
		this.alpha = alpha;
		this.runTriggerMatching = runTriggerMatching;
	}

	// get data:
	public Value.Vector getData() {
		return data;
	}

	// get data size (row number):
	public int getDataSize() {
		return dataSize;
	}

	// get variable number (only observed variables):
	public int getVariableNum() {
		return variableNum;
	}

	// get variable labels:
	public List<String> getVariableLabels() {
		return variableLabels;
	}

	// get state names of every variables:
	public ArrayList<ArrayList<String>> getStatesOfAllVariables() {
		return statesOfAllVariables;
	}

	// get the number of state of every variable:
	public int[] getStateNumOfAllVariables() {
		return stateNumOfAllVariables;
	}

	// get values of each variable (column):
	public ArrayList<ArrayList<String>> getValuesOfAllVariables() {
		return valuesOfAllVariables;
	}

	// get the dependencies matrices of current data
	public ArrayList<int[][]> getCurrentDataDependencyMatrices() {
		return currentDataDependencyMatrices;
	}

	// get whether the data matches one of the triggers:
	public boolean getMatched() {
		return isMatched;
	}

	// get the hidden model of current data (isMatched == true)
	public int[][] getCandidateHiddenModel() {
		return candidateHiddenModel;
	}

	// get the indexes of latent variable's children (isMatched == true)
	public int[] getLatentVarChildrenIndexes() {
		return latentVarChildrenIndexes;
	}

	public ArrayList<String[]> getAllInstances() {
		return allInstances;
	}

	public int getJointObservationsNum() {
		return jointObservationNum;
	}

	// Load data:
	public void loadDataFile(String path) throws IOException {
		try {
			data = Converter.load(path, false, false);
		} catch (Exception e) {
			System.out.println("Loading data failed.");
			return;
		}
		// analysis the data:
		run_analysis();
	}

	/**
	 * analysisData() contains:
	 * 
	 * 1. get variable number 2. get variable labels 3. get dependency matrices of
	 * current data 4. get probability distributions of each variable 5. get the
	 * indexed of the latent variable's children (isMatched == true) 6. get the real
	 * probability distributions of the latent variable's children (isMatched ==
	 * true) 7. get the number of joint observations
	 * 
	 * 
	 */
	public void run_analysis() throws IOException {
		// Get the data size (row number)
		dataSize = data.length();
		// Get variable number:
		variableNum = ((Value.Structured) data.elt(0)).length();

		RectangularDataSet testData = Tetrad4.cdms2TetradDiscrete(data);

		// IndependenceTest independence = new IndTestGSquare((RectangularDataSet)
		// tetData, 0.05);

		IndTestChiSquare indChi = new IndTestChiSquare((RectangularDataSet) testData, alpha);

		// Get variable labels:
		variableLabels = indChi.getVariableNames();

		// Get states and values of each variable(column):

		for (int i = 0; i < variableNum; i++) {
			ArrayList<String> unRepeatedValues = new ArrayList<String>();
			ArrayList<String> values = new ArrayList<String>();

			for (int m = 0; m < dataSize; m++) {
				values.add(data.cmpnt(i).elt(m).toString());

				boolean isRepeated = false;
				for (String value : unRepeatedValues) {
					if (value.equals(data.cmpnt(i).elt(m).toString())) {
						isRepeated = true;
						break;
					}
				}
				if (isRepeated == false) {
					unRepeatedValues.add(values.get(m));
				}
			}
			statesOfAllVariables.add(unRepeatedValues);
			valuesOfAllVariables.add(values);
		}

		// Get the number of state of each variable:
		stateNumOfAllVariables = new int[variableNum];
		for (int i = 0; i < variableNum; i++) {
			stateNumOfAllVariables[i] = statesOfAllVariables.get(i).size();
		}

		// Get current data dependency matrices:
		List<Node> nodes = indChi.getVariables();

		ArrayList<int[]> conditionVarList = getConditionVarList();
		ArrayList<int[]> remainingVarList = getRemainingVarList();

		// the null hypothesis should be the current two variables are independent

		for (int i = 0; i < remainingVarList.size(); i++) {
			if (remainingVarList.get(i).length < 2)
				continue;
			// get the current condition variables:
			int[] currentConditionVars = conditionVarList.get(i);

			// initialize a new dependency matrix:
			int[][] currentDependencyMatirx = new int[variableNum][variableNum];
			for (int n = 0; n < variableNum; n++) {
				for (int m = 0; m < variableNum; m++)
					currentDependencyMatirx[n][m] = 0;
			}

			// get a list of two variables combinations that will be test dependencies:
			ArrayList<int[]> twoVarRemainingCombinationList = getTwoVarRemainingCombinationList(
					remainingVarList.get(i));
			// build tetrad condition variable list:
			List<Node> tetradConditionVarList = new ArrayList<Node>();
			for (int n = 0; n < currentConditionVars.length; n++) {
				if (currentConditionVars[n] == 0)
					break;

				tetradConditionVarList.add(nodes.get(currentConditionVars[n] - 1));
			}

			for (int[] currentTwoVarRemainingCombination : twoVarRemainingCombinationList) {
				int firstTestNodeIndex = currentTwoVarRemainingCombination[0];
				int secondTestNodeIndex = currentTwoVarRemainingCombination[1];

				// if these two variables are dependent:
				if (indChi.isDependent(nodes.get(firstTestNodeIndex - 1), nodes.get(secondTestNodeIndex - 1),
						tetradConditionVarList) == true) {
					currentDependencyMatirx[firstTestNodeIndex - 1][secondTestNodeIndex - 1] = 1;
					currentDependencyMatirx[secondTestNodeIndex - 1][firstTestNodeIndex - 1] = 1;
				}

			}

			currentDataDependencyMatrices.add(currentDependencyMatirx);
		}

		// check whether match a trigger if required
		if(runTriggerMatching == true)
			isMatched = matchAnyTriggers();

		// get the indexed of the latent variable's children (isMatched == true):
		if (isMatched == true) {
			latentVarChildrenIndexes = new int[2];

			ArrayList<Integer> indexes = new ArrayList<Integer>();
			for (int i = 0; i < candidateHiddenModel.length; i++) {
				if (candidateHiddenModel[0][i] != 0)
					indexes.add(i - 1);
			}

			latentVarChildrenIndexes[0] = indexes.get(0);
			latentVarChildrenIndexes[1] = indexes.get(1);
		}

		// get all instances:
		for (int i = 0; i < dataSize; i++) {
			String[] instance = new String[variableNum];
			for (int n = 0; n < variableNum; n++) {
				instance[n] = valuesOfAllVariables.get(n).get(i);
			}
			allInstances.add(instance);
		}

		// get the number of different joint observations:
		ArrayList<String[]> unRepeatedInstances = new ArrayList<String[]>();
		for (String[] instance : allInstances) {
			boolean isRepeated = false;

			for (String[] uprepeatedInstance : unRepeatedInstances) {
				if (arrayEquals(uprepeatedInstance, instance)) {
					isRepeated = true;
					break;
				}
			}

			if (isRepeated == false) {
				unRepeatedInstances.add(instance);
			}
		}
		jointObservationNum = unRepeatedInstances.size();
		// System.out.println(jointObservationNum);
	}

	public boolean matchAnyTriggers() throws IOException {
		/**
		 * matchAnyTriggers() contains:
		 * 
		 * 1. check whether the current data supports any of the triggers
		 * 
		 */

		int[] firstLabel = new int[variableNum];
		for (int i = 0; i < variableNum; i++) {
			firstLabel[i] = i;
		}

		// get the hidden models (their dependency structures are the "triggers"):
		ArrayList<int[][]> hiddenModels = getHiddenModels();

		isMatched = false;
		int falseNum = (variableNum + 1) * (variableNum + 1) * (variableNum + 1);

		for (int[][] currentHiddenModel : hiddenModels) {

			GetFullLablesCombinationList gcHidden = new GetFullLablesCombinationList(currentHiddenModel, true);
			ArrayList<int[][]> allHiddenLabelsCombinationDAGList = gcHidden.getLablesCombinationDAGList();

			for (int[][] currentHiddenCombinationDAG : allHiddenLabelsCombinationDAGList) {
				// get the dependency matrices of current hidden label combination model,
				// "true" means there is a hidden variable in the model:
				ArrayList<int[][]> currentHiddenModelDependencyMatrices = generateDSeparationDependencyMatrices(
						currentHiddenCombinationDAG, variableNum + 1, true);

				int incorrects = compareHiddenAndObservedDependencyMetrices(currentDataDependencyMatrices,
						currentHiddenModelDependencyMatrices);

				if (incorrects < falseNum) {
					candidateHiddenModel = new int[variableNum + 1][variableNum + 1];
					candidateHiddenModel = currentHiddenCombinationDAG.clone();
					falseNum = incorrects;
					int totalEntry = (variableNum + 1) * (variableNum + 1) * currentDataDependencyMatrices.size();
					if (incorrects < totalEntry * errorRate)
						isMatched = true;
				}

			}

		}

		return isMatched;
	}

	// public boolean

	private ArrayList<int[][]> generateDSeparationDependencyMatrices(int[][] DAG, int varNum,
			boolean containHiddenVar) {
		ArrayList<int[][]> dependencyMatrices = new ArrayList<int[][]>();

		GenerateAllDependencies gd = new GenerateAllDependencies(DAG, varNum, containHiddenVar);

		dependencyMatrices = gd.getResult();

		return dependencyMatrices;
	}

	private ArrayList<int[]> getConditionVarList() {

		/**
		 * Generate all possible condition variable set (by setting currentVarNum to
		 * varNum):
		 * 
		 * For example:
		 * 
		 * For variable A, B, C in a given DAG, the condition variable set are: Ø(0), 1,
		 * 2, 3, 12, 13, 23, 123
		 * 
		 * If the current DAG contains a hidden variable (the first variable), then in
		 * this case, the condition variable set are: Ø(0), 2, 3, 4, 23, 24, 24, 234
		 * 
		 */
		// because we dont condition on the hidden variable, thus varNum should
		// change to varNum - 1:

		ArrayList<int[]> resultList = new ArrayList<int[]>();

		int[] iArr = new int[variableNum];

		// Set the empty variable set Ø by 0
		int[] fisrtEmptySet = new int[1];
		fisrtEmptySet[0] = 0;
		resultList.add(fisrtEmptySet);

		for (int tempNum = 1; tempNum < variableNum + 1; tempNum++) {
			int num = 0;
			int pos = 0;
			int ppi = 0;

			for (;;) {
				if (num == variableNum) {
					if (pos == 1)
						break;

					pos -= 2;
					ppi = iArr[pos];
					num = ppi;
				}

				iArr[pos++] = ++num;

				if (pos != tempNum)
					continue;

				int[] tempArray = new int[pos];

				for (int i = 0; i < pos; i++) {

					tempArray[i] = iArr[i];
				}

				resultList.add(tempArray);
			}
		}

		return resultList;
	}

	private ArrayList<int[]> getRemainingVarList() {
		/**
		 * Generate the remaining variable list which will be applied D-separation
		 * rules.
		 * 
		 * For example:
		 * 
		 * for a 4 variable matrix
		 * 
		 * condition remaining Ø(0) 1, 2, 3, 4 1 2, 3, 4 2 1, 3, 4 3 1, 2, 4 4 2, 3, 4
		 * 1, 2 3, 4 1, 3 2, 4 1, 4 2, 3 2, 3 1, 4 2, 4 1, 3 3, 4 1, 2 1, 2, 3 4 1, 2, 4
		 * 3 1, 3, 4 2 2, 3, 4 1 1, 2, 3, 4 Ø(0)
		 */

		ArrayList<int[]> resultList = new ArrayList<int[]>();

		ArrayList<int[]> conditionOnVarList = getConditionVarList();

		// The first remaining variable list should be the full list:
		int[] firstResult = new int[variableNum];
		for (int i = 0; i < variableNum; i++) {
			firstResult[i] = i + 1;
		}
		resultList.add(firstResult);

		int num = 0;

		for (int[] currentConditionOnVarList : conditionOnVarList) {

			// cuz the first list from resultList is null(0), so skip it.
			if (num == 0) {
				num++;
				continue;
			}

			ArrayList<Integer> fullVarArrayList = new ArrayList<Integer>();

			for (int i = 1; i <= variableNum; i++) {
				fullVarArrayList.add(i);
			}

			for (int var : currentConditionOnVarList) {
				for (int i = 0; i < fullVarArrayList.size(); i++) {
					// remove the variables from fullVarArrayList because they exist
					// in currentConditionOnVarList
					if (fullVarArrayList.get(i) == var)
						fullVarArrayList.remove(i);
				}
			}

			int[] tempResult = new int[fullVarArrayList.size()];
			// transfer the ArrayList to int[]:
			for (int i = 0; i < tempResult.length; i++) {
				tempResult[i] = fullVarArrayList.get(i);
			}

			if (tempResult.length == 0) {
				int[] finalResult = new int[1];
				finalResult[0] = 0;
				resultList.add(finalResult);
			} else
				resultList.add(tempResult);

		}

		return resultList;
	}

	private ArrayList<int[]> getTwoVarRemainingCombinationList(int[] currentRemainList) {
		/**
		 * Get all combination of two variables that will be applied D-Separation rules
		 * to check their dependencies.
		 * 
		 * For example: If the current remaining variables are 1, 2, 3 So all the
		 * combinations are:
		 * 
		 * 1,2 1,3 2,3
		 */

		ArrayList<int[]> result = new ArrayList<int[]>();

		for (int n = 0; n < currentRemainList.length; n++) {
			for (int m = n + 1; m < currentRemainList.length; m++) {
				int[] temp = new int[2];
				temp[0] = currentRemainList[n];
				temp[1] = currentRemainList[m];

				int num1 = currentRemainList[n];
				int num2 = currentRemainList[m];

				result.add(temp);
			}
		}

		return result;
	}

	private ArrayList<int[][]> getHiddenModels() throws IOException {
		
		ArrayList<int[][]> hiddenModels = new ArrayList<int[][]>();
		/*
		 * We run the trigger discovery program in advance and save all triggers in a
		 * local txt file. Then read from the local txt to get all triggers.
		 */

//		triggerIO io = new triggerIO(4 + 1);
//
//		hiddenModels = io.readtxt("4var_triggers");
		
		// as there is only two triggers, so we just do hard code here
		/**
		 *  trigger1 (big W):
         *    00011
         *    00101
         *    00010
         *    00000
         *    00000
         *    
		 *  trigger2 (big W with one extra arc):
		 *    00011
         *    00001
         *    00010
         *    00000
         *    00000
         *  
		 * */
		
		int[][] trigger1 = new int[5][5];
		for(int n=0; n<trigger1.length; n++)
		{
			for(int m=0; m<trigger1.length; m++)
				trigger1[n][m] = 0;
		}
		trigger1[0][3] = 1;
		trigger1[0][4] = 1;
		trigger1[1][2] = 1;
		trigger1[1][4] = 1;
		trigger1[2][3] = 1;
		
		
		int[][] trigger2 = new int[5][5];
		for(int n=0; n<trigger2.length; n++)
		{
			for(int m=0; m<trigger2.length; m++)
				trigger2[n][m] = 0;
		}
		trigger2[0][3] = 1;
		trigger2[0][4] = 1;
		trigger2[1][4] = 1;
		trigger2[2][3] = 1;
		
		hiddenModels.add(trigger1);
		hiddenModels.add(trigger2);
				
		return hiddenModels;
	}

	private ArrayList<int[][]> reduceHiddenModelDependencyMetricesDimension(
			ArrayList<int[][]> hiddenDependencyMatrices) {
		/**
		 * Because we do not care about the dependency relationships between the hidden
		 * variable and other observed variables, so we should remove the dimension of
		 * hidden variable in the dependency matrices in order to compare with observed
		 * dependency matrices.
		 * 
		 * For example:
		 * 
		 * H 1 2 3 1 2 3 --------- ------- H| 0 1 0 1 -------> 1| 0 1 0 1| 1 0 1 0 2| 1
		 * 0 1 2| 0 1 0 1 3| 0 1 0 3| 1 0 1 0
		 * 
		 */

		ArrayList<int[][]> resultList = new ArrayList<int[][]>();

		for (int[][] currentMatrix : hiddenDependencyMatrices) {
			int[][] temp = new int[variableNum][variableNum];
			for (int n = 0; n < variableNum; n++) {
				for (int m = 0; m < variableNum; m++)
					temp[n][m] = currentMatrix[n + 1][m + 1];
			}
			resultList.add(temp);
		}

		return resultList;
	}

	private int compareHiddenAndObservedDependencyMetrices(ArrayList<int[][]> observedDependencyMatrices,
			ArrayList<int[][]> hiddenDependencyMatrices) {
		// reduce the dimension of the hidden model matrices because we dont care the
		// dependencies between the hidden variable and other observed variables:
		ArrayList<int[][]> reducedHiddenDependencyMatrices = reduceHiddenModelDependencyMetricesDimension(
				hiddenDependencyMatrices);

		boolean isSame = true;

		int inCorrects = 0;

		for (int i = 0; i < observedDependencyMatrices.size(); i++) {
			int[][] currentObservedDependencyMatrix = observedDependencyMatrices.get(i);
			int[][] currentHiddenDependencyMatrix = reducedHiddenDependencyMatrices.get(i);

			for (int n = 0; n < currentObservedDependencyMatrix.length; n++) {
				for (int m = 0; m < currentObservedDependencyMatrix[n].length; m++) {
					if (currentObservedDependencyMatrix[n][m] != currentHiddenDependencyMatrix[n][m]) {
						isSame = false;
						inCorrects++;
						// break;
					}
				}

				// if(isSame == false)
				// break;
			}

			// if(isSame == false)
			// break;
		}

		int totalEntryNum = variableNum * variableNum * hiddenDependencyMatrices.size();

		// if the incorrect number is less than 5% of the total entry number, than
		// return true
		// if(inCorrects <= totalEntryNum*0.15)
		// isSame = true;

		return inCorrects;
	}

	// Get the adjacent matrix (maybe it is not a upper-triangle matrix) of the
	// hidden model
	// for the current data if it matches any trigger:
	public int[][] getHiddenModelForCurrentData() {
		int[][] hiddenModel = new int[variableNum + 1][variableNum + 1];

		// if(isMatched == false)
		// System.out.println("The current data does not match any trigger...");
		if (isMatched == true)
			hiddenModel = candidateHiddenModel.clone();

		return hiddenModel;
	}

	// Check whether two arrays are equal or not
	private boolean arrayEquals(Object[] o1, Object[] o2) {
		if (o1.length != o2.length)
			return false;

		for (int i = 0; i < o1.length; i++) {

			if (!o1[i].equals(o2[i]))
				return false;
		}

		return true;
	}

}
