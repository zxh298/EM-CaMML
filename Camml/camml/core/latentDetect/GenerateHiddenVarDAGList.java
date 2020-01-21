package camml.core.latentDetect;

import java.util.ArrayList;

/**
 *  This class replace a single arc with a hidden common cause 
 *  from a given DAG, which does not contain any hidden variables
 *  
 *  Input: a given list of DAGs
 *  Output: a corresponding list of DAGs (with hidden common cause)
 *  
 *  Example:
 *  
 *  A->B->C<-D  to  A->H<-B->C<-D
 *  
 * */

public class GenerateHiddenVarDAGList {

	ArrayList<int[][]> observedDAGList = new ArrayList<int[][]>();

	static int varNum;

	public GenerateHiddenVarDAGList(ArrayList<int[][]> observedDAGList, int varNum)
	{
		this.observedDAGList = observedDAGList;
		this.varNum = varNum;
	}

	public static ArrayList<int[][]> makeHiddenMatrix(int[][] observedDAG)
	{
		// One observed can have multiple corresponding hidden matrix (the number equals to its arc number)
		ArrayList<int[][]> resultList = new ArrayList<int[][]>();

		int[][] hiddenMatrix = new int[varNum+1][varNum+1];

		/**
		 *  Adding a new row and a new column to the current observed matrix,
		 *  because of inserting a hidden variable (to be the first variable):
		 *  n: row number; m: column number
		 *  
		 *  Example:
		 *    1 2 3               H 1 2 3
		 *  -------              --------
		 *  1|0 1 1             H|0 0 0 0
		 *  2|0 0 1    -------> 1|0 0 1 1
		 *  3|0 0 0             2|0 0 0 1
		 *                      3|0 0 0 0 
		 * 
		 * */
		for(int n = 0; n < varNum + 1; n++)
		{
			for (int m = 0; m < varNum + 1; m++)
			{
				if (n == 0 || m == 0)
					hiddenMatrix[n][m] = 0;
				else
					hiddenMatrix[n][m] = observedDAG[n-1][m-1];			
			}

		}

//		System.out.println("hidden matrix:");
//		for(int n = 0; n < varNum + 1; n++)
//		{
//			for (int m = 0; m < varNum + 1; m++)
//			{ 
//				System.out.print(hiddenMatrix[n][m] + " ");
//
//			}
//
//			System.out.println();
//		}
//
//		System.out.println("*******************");

		/**
		 * Replace a single arc with a hidden common cause
		 * 
		 * Example:
		 * 
		 * 0 0 0 0             0 1 1 0
		 * 0 0 1 1   ------>   0 0 0 1
		 * 0 0 0 1             0 0 0 1
		 * 0 0 0 0             0 0 0 0
		 * 
		 * */

		for(int n = 1; n < varNum + 1; n++)
		{
			for(int m = 1; m < varNum + 1; m++)
			{
				int[][] tempHiddenMatrix = new int[varNum+1][varNum+1];

				tempHiddenMatrix = cloneMatrix(hiddenMatrix, varNum+1);

				if (tempHiddenMatrix[n][m] == 1)
				{
					tempHiddenMatrix[0][n] = 1;
					tempHiddenMatrix[0][m] = 1;
					tempHiddenMatrix[n][m] = 0;
					resultList.add(tempHiddenMatrix);
				}
			}
		}

		return resultList;
	}

	private static int[][] cloneMatrix(int[][] matrix, int varNum)
	{
		int[][] clone = new int[varNum][varNum];

//		for(int n = 0; n < varNum; n++)
//		{
//		    for(int m = 0; m < varNum; m++)
//		    {
//		    	clone[n][m] = matrix[n][m];
//		    }
//		}

		for(int i=0;i<matrix.length;i++)
		{
			clone[i]=matrix[i].clone();
		}

		return clone;
	}

	public static void main(String[] args)
	{
		int[][] matrix = new int[4][4];

		for(int n = 0; n < 4; n++)
		{
			for(int m = 0; m < 4; m++)
			{
				matrix[n][m] = 0;
			}
		}

		matrix[0][1] = 1;
		matrix[0][2] = 1;
		matrix[0][3] = 1;
		matrix[1][2] = 1;
		matrix[1][3] = 1;
		matrix[2][3] = 1;

		GenerateHiddenVarDAGList.varNum = 4;
		ArrayList<int[][]> resultList = GenerateHiddenVarDAGList.makeHiddenMatrix(matrix);

		System.out.println("Size: " + resultList.size());

		for(int[][] temp : resultList)
		{
			for(int n = 0; n < varNum + 1; n++)
			{
				for(int m = 0; m < varNum + 1; m++)
				{
					System.out.print(temp[n][m] + " ");
				}

				System.out.println();
			}

			System.out.println("*********************");
		}
	}
}
