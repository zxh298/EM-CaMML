package camml.core.latentDetect;

import java.util.ArrayList;

public class GetFullLablesCombinationList {

	/** the total number */  
	private int total = 0;  
	/** the number of calling allRank*/  
	private int call  = 0;  

	ArrayList<int[]> labelCombinationresultList = new ArrayList<int[]>();

	private int[] array;

	private int length;

	private int[][] currentMatrix;

	private boolean hasHiddenVar;

	public GetFullLablesCombinationList(int[] currentArray, int length)
	{
		this.length = length;
		array = new int[length];

		for(int i = 0; i < currentArray.length; i++)
		{
			array[i] = currentArray[i];
		}
	}


	public GetFullLablesCombinationList(int[][] currentMatrix, boolean hasHiddenVar)
	{
		length = currentMatrix.length;

		this.hasHiddenVar = hasHiddenVar;

		this.currentMatrix = currentMatrix.clone();
	}

	public ArrayList<int[][]> getLablesCombinationDAGList()
	{
		ArrayList<int[][]> resultList = new ArrayList<int[][]>();

		// if there is no hidden variable:
		if(hasHiddenVar == false)
		{
			array = new int[currentMatrix.length];
			for(int i = 0; i < currentMatrix.length; i++)
			{
				array[i] = i;
			}

			allRank(array, 0, currentMatrix.length);

			for(int[] currentLabelCombination : labelCombinationresultList)
			{
				int[][] tempObservedDAG = new int[currentMatrix.length][currentMatrix.length];

				for(int n = 0; n < currentMatrix.length; n++)
				{
					for(int m = 0; m < currentMatrix[n].length; m++)
					{
						tempObservedDAG[n][m] = currentMatrix[currentLabelCombination[n]][currentLabelCombination[m]];						
					}
				}

				resultList.add(tempObservedDAG);
			}
		}

		if(hasHiddenVar == true)
		{
			// store the hidden variable locations:
			int varHiddenConnectTo1 = 0;

			int varHiddenConnectTo2 = 0;

			int flag = 0;

			for(int i = 0; i < currentMatrix.length; i++)
			{
				if(currentMatrix[0][i] == 1)
				{
					if(flag == 0)
					{
						varHiddenConnectTo1 = i;
						flag ++;
						continue;
					}
					if(flag > 0)
						varHiddenConnectTo2 = i;
				}
			}

			array = new int[currentMatrix.length - 1];

			for(int i = 0; i < currentMatrix.length - 1; i++)
			{
				array[i] = i;
			}

			allRank(array, 0, currentMatrix.length - 1);

			for(int[] currentLabelCombination : labelCombinationresultList)
			{

				for(int i = 0; i < currentLabelCombination.length; i++)
				{
					currentLabelCombination[i] = currentLabelCombination[i] + 1;
				}


				// initialize the temp matrix:
				int[][] tempHiddenDAG = new int[currentMatrix.length][currentMatrix.length];				
				for(int n = 0; n < currentMatrix.length; n++)
				{
					for(int m = 0; m < currentMatrix[n].length; m++)
						tempHiddenDAG[n][m] = 0;
				}

				// leave the first row, and deal with it later:
				for(int n = 1; n < currentMatrix.length; n++)
				{
					for(int m = 1; m < currentMatrix[n].length; m++)
					{
						tempHiddenDAG[n][m] = currentMatrix[currentLabelCombination[n-1]][currentLabelCombination[m-1]];						
					}
				}

				for(int i = 0; i < currentLabelCombination.length; i++)
				{
					if(currentLabelCombination[i] == varHiddenConnectTo1 || currentLabelCombination[i] == varHiddenConnectTo2)
						tempHiddenDAG[0][i+1] = 1;						
				}

				// deal with the first row (relocate hidden variable):
				//				tempHiddenDAG[0][currentLabelCombination[varHiddenConnectTo1 - 2]] = 1;
				//				tempHiddenDAG[0][currentLabelCombination[varHiddenConnectTo2 - 2]] = 1;
				//				System.out.println("cc: " + varHiddenConnectTo1);
				//				System.out.println("cc: " + varHiddenConnectTo2);
				//				System.out.println("cc1: " + currentLabelCombination[varHiddenConnectTo1-1]);
				//				System.out.println("cc2: " + currentLabelCombination[varHiddenConnectTo2-1]);
				resultList.add(tempHiddenDAG);
			}



			//			for(int[] currentLabelCombination : labelCombinationresultList)
			//			{
			//				int[][] tempObservedDAG = new int[currentMatrix.length][currentMatrix.length];
			//
			//				for(int n = 0; n < currentMatrix.length; n++)
			//				{
			//					for(int m = 0; m < currentMatrix[n].length; m++)
			//					{
			//						tempObservedDAG[n][m] = currentMatrix[currentLabelCombination[n]][currentLabelCombination[m]];						
			//					}
			//				}
			//
			//				resultList.add(tempObservedDAG);
			//			}



		}



		return resultList;
	}

	public ArrayList<int[]> getLabelsCombinationResult()
	{
		allRank(array, 0, length);
		return labelCombinationresultList;
	}

	private void store(int[] array) {  

		int[] temp = new int[array.length];

		for (int i = 0; i < array.length; i++)
		{
			temp[i] = array[i];
		}

		labelCombinationresultList.add(temp);
		total++;
	}  

	private void swap(int[] array, int a, int b) {  
		if (a == b) {  
			return;  
		}  
		array[a] = array[a] ^ array[b];  
		array[b] = array[a] ^ array[b];  
		array[a] = array[a] ^ array[b];  
	}  

	/** 
	 * the first unit of array is fixed<br> 
	 * the remaining units do allRank(array, begin + 1, length); <br> 
	 * of course, it has to let every unit be the first one
	 *  
	 * @param array 
	 * @param begin 
	 * @param length 
	 */  
	private void allRank(int[] array, int begin, int length) {  
		call++;  
		if (begin == length - 1) {  
			// the end of every recursive, call store  
			store(array);   
		} else {  
			for (int i = begin; i < length; i++) {  
				//do swap only there is no repeat from begin~(i-1)th units
				boolean doubleSign = false;  
				for(int j = begin;j<i;j++){  
					if(array[j] == array[i]){  
						doubleSign = true;  
					}  
				}  
				if(doubleSign){  
					continue;  
				}  
				// let every unit be the first one using for
				swap(array, begin, i);  
				// ignore the first one, do recursive from the second 
				allRank(array, begin + 1, length);  
				// if we do not return to the original environment, then
				// allRank will be a function that may change the values
				// of array, then using “for(int i = begin; i < length; i++)"
				// to let every unit to be the first one will not function.
				swap(array, i, begin);  
			}  
		}  
	} 


	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int[][] testMatrix = new int[5][5];

		for(int n = 0; n < 5; n++)
		{
			for(int m = 0; m < 5; m++)
				testMatrix[n][m] = 0;
		}

		testMatrix[0][3] = 1;
		testMatrix[0][4] = 1;
		testMatrix[1][2] = 1;
		testMatrix[1][3] = 1;
		testMatrix[2][4] = 1;

		GetFullLablesCombinationList gc = new GetFullLablesCombinationList(testMatrix, true);

		ArrayList<int[][]> resultList = gc.getLablesCombinationDAGList();

		//System.out.println("Size: " + resultList.size());

		for(int[][] currentMatrix : resultList)
		{
			for(int n = 0; n < currentMatrix.length; n++)
			{
				for(int m = 0; m < currentMatrix.length; m++)
				{
					System.out.print(currentMatrix[n][m] + " ");
				}

				System.out.println();
			}

			System.out.println("**************");
		}
	}

}
