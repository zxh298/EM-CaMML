package camml.core.latentDetect;

import java.util.ArrayList;

public class DAGList {
	
	// the number of vertices:
	protected int varNum;

	//protected int[][] fullConnectedArcMatrix;

	// the list of arc matrices for the current vertices number 
	protected ArrayList<int[][]> matrixList = new ArrayList<int[][]>();

	public ArrayList<int[][]> getResultList()
	{
		return matrixList;
	}
	
    public DAGList(int varNum)
	{
		this.varNum = varNum;
	}

	// get all possible DAGs(stored in a ArrayList):
	public void makeMatrixList()
	{
		// get the fully connected matrix as the first one:
		//matrixList.add(fullConnectedArcMatrix);
        // the maximum number of arcs:
		int maximumNumOfArc = varNum * (varNum - 1) / 2;
		// the minimum number of arcs should be varNum - 1, or there will be vertices that are not connected: 
		int minimumNumOfArc = varNum - 1;
	    //log.debug("minimum number of arcs for " + varNum + "variables: " + minimumNumOfArc + "/n");
		// the maximum number of 0 in the upper triangle should be no more than n(n-1)/2 - (n-1)
		int maxZeroNum = varNum*(varNum - 1)/2 - varNum + 1;
		//log.debug("maximum number of 0 in matrix for " + varNum + "variables: " + maxZeroNum + "/n");
		
		//GenerateAllDAGs ga = new GenerateAllDAGs();
		//getAllDAGs(int maxZeroNum, int maximumNumOfArc, int varNum)
		matrixList = GenerateAllDAGs.getAllDAGs(maxZeroNum, maximumNumOfArc, varNum);
		
	}

//	public ArrayList<int[][]> getMatrixList()
//	{
//		return matrixList;
//	}

	public int getMatrixListLength()
	{
		return matrixList.size();
	}
	
	public void print()
	{
		System.out.println("For " + varNum + " variables:");
		System.out.println("There are " + matrixList.size() + " possible DAGs:");
        System.out.println();
        System.out.println("***********************");
		
		for (int[][] array : matrixList)
		{
			for (int n = 0; n < varNum; n++)
			{
				for (int m = 0; m < varNum; m++)
				{
					System.out.print(array[n][m] + " ");
				}
				
				System.out.println();
			}
			
			System.out.println("*************************");
		}
		
		System.out.println();
		System.out.println("There are " + matrixList.size() + " possible DAGs:");
	}
	
	// convert a two dimension matrix to a one dimension matrix:
	private int[] convertToOneDimensionMatrix(int[][] twoDimensionMatrix, int varNum)
	{
	    // twoDimensionMatrix: the two dimension matrix that you want to convert
		// varNum: the size (equals to the row or the column length) of the two dimension matrix 
		
		// length: the total number of units of the two dimension matrix that you want to convert:
		int length = varNum * varNum;
		
		// tempArrayList: store the result in the format of ArrayList, then convert it to int[]:
		ArrayList <Integer> tempArrayList = new ArrayList<Integer>();
		
		// result (one dimensional)
		int[] result = new int[length];
		
		for(int m = 0; m < varNum; m++)
		{
			for(int n = 0; n < varNum; n++)
			{
				tempArrayList.add(twoDimensionMatrix[m][n]);
			}
		}
		// convert the tempArrayList to int[]:
		for (int i = 0; i < tempArrayList.size(); i++)
		{
			result[i] = tempArrayList.get(i);
		}
		
		return result;
		
	}
	// convert a one dimension matrix to a two dimension matrix:
	private int[][] convertToTwnDimensionMatrix(int[] oneDimensionMatrix, int varNum)
	{
		// oneDimensionMatrix: the one dimension matrix that you want to convert:
		// matrixSize: the size (equals to the row or the column length) of the two dimension matrix of the result
		int[][] result = new int[varNum][varNum];
		
		for (int m = 0; m < varNum; m++)
		{
			for (int n = 0; n < varNum; n++)
			{
				result[m][n] = oneDimensionMatrix[m*varNum + n];	
			}
			
		}
		
	    return result;
	}
    
}
