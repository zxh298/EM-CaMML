package camml.core.latentDetect;

import java.util.ArrayList;

public class CompareDependencyMatrices {

	/**
	 * Compare the dependency matrices of hidden models (including a single hidden variable as 
	 * a common cause) and observed models (without any hidden variables):
	 * 
	 * 1) If they are totally same, it means that all their dependencies in terms of the same set
	 *    of condition variable are same (involving the hidden variable does not make any change).
	 * 
	 * 2) If they are different, it means that involving the hidden variable has provided different
	 *    dependency relationships among the remaining observed variables in the model, and thus we
	 *    should keep this hidden variable.
	 * 
	 * */

	private ArrayList<int[][]> hiddenDependencyMatrices = new ArrayList<int[][]>();
	private ArrayList<ArrayList<int[][]>> allObservedDAGsDependencyMatrices = new ArrayList<ArrayList<int[][]>>();
	private int observedVarNum;
    // If there exists observed models with the same dependencies as the current hidden model,
	// store their index:
	//private ArrayList<Integer> indexOfObservedModelsHaveSameDependencies = new ArrayList<Integer>();

	public CompareDependencyMatrices(ArrayList<int[][]> hiddenDependencyMatrices, 
			ArrayList<ArrayList<int[][]>> allObservedDAGsDependencyMatrices, int observedVarNum)
	{
		this.hiddenDependencyMatrices = (ArrayList<int[][]>) hiddenDependencyMatrices.clone();
		this.allObservedDAGsDependencyMatrices = (ArrayList<ArrayList<int[][]>>) allObservedDAGsDependencyMatrices.clone();
		this.observedVarNum = observedVarNum;
	}

	private ArrayList<int[][]> reduceMetricesDimension()
	{
		/**
		 *  Because we do not care about the dependency relationships between the hidden variable 
		 *  and other observed variables, so we should remove the dimension of hidden variable in
		 *  the dependency matrices in order to compare with observed dependency matrices.
		 * 
		 *  For example:
		 *  
		 *    H 1 2 3                  1 2 3 
		 *   ---------                -------
		 * H| 0 1 0 1     ------->  1| 0 1 0 
		 * 1| 1 0 1 0               2| 1 0 1
		 * 2| 0 1 0 1               3| 0 1 0
		 * 3| 1 0 1 0
		 * 
		 * */

		ArrayList<int[][]> resultList = new ArrayList<int[][]>();

		for(int[][] currentMatrix : hiddenDependencyMatrices)
		{
			int[][] temp = new int[observedVarNum][observedVarNum];
			for(int n = 0; n < observedVarNum; n++)
			{
				for(int m = 0; m < observedVarNum; m++)
					temp[n][m] = currentMatrix[n+1][m+1];
			}
			resultList.add(temp);
		}

		return resultList;
	}

	public boolean hiddenAndObeservedDependencySame()
	{

		ArrayList<int[][]> reducedHiddenDependencyMatrices = reduceMetricesDimension();

		/**
		 * numberOfPbservedDAGsHaveDifferentDependency: 
		 * The total number of observed DAGs that have the same dependency 
		 * structure as the current hidden DAG.
		 * */
		int differentNum = 0;
		
		for(int num = 0; num < allObservedDAGsDependencyMatrices.size(); num++)
		{
			boolean isSame = true;				
			//Get the dependency matrices list of current num(th) observed DAG:
			ArrayList<int[][]> currentObservedDependencyMatrices = allObservedDAGsDependencyMatrices.get(num);

			for(int i = 1; i < reducedHiddenDependencyMatrices.size(); i++)
			{
				int[][] currentHiddenMatrix = reducedHiddenDependencyMatrices.get(i);
				int[][] currentObservedMatrix = currentObservedDependencyMatrices.get(i);

				for(int n = 0; n < currentObservedMatrix.length; n++)
				{
					for(int m = 0; m < currentObservedMatrix[n].length; m++)
					{
						// even there is only one unit is different, the result will be false
						if(currentHiddenMatrix[n][m] != currentObservedMatrix[n][m])		    			
						{ 
							isSame = false;
							break;
						}		    			
					}

					if(isSame == false)		    		
						break;
				}
								
				if(isSame == false)
				{
					differentNum++;
					break;
				}
			}    		
		
//			if(isSame == true)
//			{
//				indexOfObservedModelsHaveSameDependencies.add(num);
//			}
		}
		
		if(differentNum == allObservedDAGsDependencyMatrices.size())
			return false;
        
		return true;
	}

//    public ArrayList<Integer> getObservedIndexWithSameDependencies()
//    {
//    	return indexOfObservedModelsHaveSameDependencies;
//    }

	public static void main(String[] args)
	{
		ArrayList<int[][]> observedMatrices = new ArrayList<int[][]>();
		ArrayList<int[][]> hiddenMatrices = new ArrayList<int[][]>();

		int[][] observedMatrix1 = new int[3][3];
		int[][] observedMatrix2 = new int[3][3];
		int[][] observedMatrix3 = new int[3][3];
		int[][] hiddenMatrix1 = new int[4][4];
		int[][] hiddenMatrix2 = new int[4][4];
		int[][] hiddenMatrix3 = new int[4][4];

		for(int n = 0; n < 3; n++)
		{
			for(int m = 0; m < 3; m++)
			{
				observedMatrix1[n][m] = 0;
				observedMatrix2[n][m] = 0;
				observedMatrix3[n][m] = 0;			
			}
		}

		for(int n = 0 ; n < 4; n++)
		{
			for(int m = 0; m < 4; m++)
			{
				hiddenMatrix1[n][m] = 0;
				hiddenMatrix2[n][m] = 0;
				hiddenMatrix3[n][m] = 0;
			}
		}

		/**
		 *   observed:
		 * 
		 *   0 1 0    0 0 1     0 1 0 
		 *   1 0 0    0 0 1     1 0 1
		 *   0 0 0    1 1 0     0 1 0
		 *    
		 *   hidden:
		 *   
		 *   0 1 0 0   0 0 1 1    0 0 1 0   
		 *   1 0 1 0   0 0 0 1    0 0 1 0
		 *   0 1 0 0   1 0 0 1    1 1 0 1
		 *   0 0 0 0   1 1 1 0    0 0 1 0
		 *   
		 * */

		observedMatrix1[0][1] = 1;
		observedMatrix1[1][0] = 1;
		observedMatrix2[0][2] = 1;
		observedMatrix2[1][2] = 1;
		observedMatrix2[2][0] = 1;
		observedMatrix2[2][1] = 1;
		observedMatrix3[0][1] = 1;
		observedMatrix3[1][0] = 1;
		observedMatrix3[1][2] = 1;
		observedMatrix3[2][1] = 1;

		hiddenMatrix1[0][1] = 1;
		hiddenMatrix1[1][0] = 1;
		hiddenMatrix1[1][2] = 1;
		hiddenMatrix1[2][1] = 1;
		hiddenMatrix2[0][2] = 1;
		hiddenMatrix2[0][3] = 1;
		hiddenMatrix2[1][3] = 1;
		hiddenMatrix2[2][0] = 1;
		//hiddenMatrix2[2][3] = 1;
		hiddenMatrix2[3][0] = 1;
		hiddenMatrix2[3][1] = 1;
		hiddenMatrix2[3][2] = 1;
		hiddenMatrix3[0][2] = 1;
		hiddenMatrix3[1][2] = 1;
		hiddenMatrix3[2][0] = 1;
		hiddenMatrix3[2][1] = 1;
		hiddenMatrix3[2][3] = 1;
		hiddenMatrix3[3][2] = 1;

		observedMatrices.add(observedMatrix1);
		observedMatrices.add(observedMatrix2);
		observedMatrices.add(observedMatrix3);

		hiddenMatrices.add(hiddenMatrix1);
		hiddenMatrices.add(hiddenMatrix2);
		hiddenMatrices.add(hiddenMatrix3);

		//		CompareHiddenAndObservedDependencyMatrices com = new CompareHiddenAndObservedDependencyMatrices(observedMatrices,
		//				hiddenMatrices, 3);

		//		ArrayList<int[][]> reducedHiddenMatrices = com.reduceMetricesDimension();
		//		
		//		for(int[][] currentHiddenMatrix : reducedHiddenMatrices)
		//		{
		//			for(int n = 0; n < 3; n++)
		//			{
		//				for(int m = 0; m < 3; m++)
		//				{
		//					System.out.print(currentHiddenMatrix[n][m]+" ");
		//				}
		//				System.out.println();
		//			}
		//			System.out.println("*********************");
		//		}
		//		
		////		
		//		boolean isSame = com.hiddenAndObeservedDependencySame();
		//		
		//		System.out.println(isSame);


	}

}
