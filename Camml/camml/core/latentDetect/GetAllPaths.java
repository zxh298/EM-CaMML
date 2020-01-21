package camml.core.latentDetect;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

public class GetAllPaths {

    private static int[][] matrix;
    private static int varNum;
	
	private static int START;
	private static int END;
	
	// acrDirection: to specify whether the direction of 
	//               each arc matters.
	// False: generate all paths between two nodes
	// True: generate all descendants of a given node (the direction of arcs deciding parent-descendant relationship)
	
	private static boolean arcDirection;
    private ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
   // private static ArrayList<LinkedList<Integer>> secondResult = new ArrayList<LinkedList<Integer>>();
    
	// get all paths from one variable to another
    // flag: true---care about the arc direction
    //       false-- do not care about the arc direction
	public GetAllPaths(int[][] currentMatrix, int currentVarNum, int startVar, int endVar, boolean flag)
	{
		START = startVar;
		END = endVar;
		varNum = currentVarNum;
		
		matrix = new int[varNum][varNum];
		for(int n = 0; n < varNum; n++)
		{
			for(int m = 0; m < varNum; m++)
			{
				matrix[n][m] = currentMatrix[n][m];
			}
		}
		
		arcDirection = flag;
	}

	private void breadthFirst(MatrixHashMap graph, LinkedList<Integer> visited) {
		LinkedList<Integer> nodes = graph.adjacentNodes(visited.getLast());
		// examine adjacent nodes
		for (Integer node : nodes) {
			if (visited.contains(node)) {
				continue;
			}
			if (node.equals(END)) {
				visited.add(node);
				storeCurrentVisited(visited);
				//result.add(visited);
				visited.removeLast();
				break;
			}
		}
		// in breadth-first, recursion needs to come after visiting adjacent nodes
		for (Integer node : nodes) {
			// if the visited already contains this node or the current node is the END node: continue
			if (visited.contains(node) || node.equals(END)) {
				continue;
			}
			visited.addLast(node);
			breadthFirst(graph, visited);
			visited.removeLast();
		}
	}

	private void storeCurrentVisited(LinkedList<Integer> visited) {
	
		ArrayList<Integer> temp = new ArrayList<Integer>();
		
		for (Integer node : visited) {
			temp.add(node);
		}
		
		result.add(temp);
	}
	
	public ArrayList<ArrayList<Integer>> getResult()
	{
				
		GetAllPaths.MatrixHashMap map = new MatrixHashMap();

		map.transferMatrixToHashMap();

		LinkedList<Integer> visited = new LinkedList();
		visited.add(START);
		breadthFirst(map, visited);
		
		
		return result;
	}
	
//	public ArrayList<LinkedList<Integer>> getSecondResult()
//	{
//		return secondResult;
//	}

	// *********************** Inner class: Graph **************************

	private static class MatrixHashMap
	{
		private Map<Integer, LinkedHashSet<Integer>> map = new HashMap();

//		private int[][] matrix; 
//		private int varNum;


		public void transferMatrixToHashMap()
		{
			for (int n = 0; n < varNum; n++)
			{
				for (int m = 0; m < varNum; m++)
				{
					if (matrix[n][m] == 1)
						{
						   if (arcDirection == false)
						      addAdjacency(n+1, m+1);
						   if (arcDirection == true)
							  addEdge(n+1, m+1);
						}
				}
			}
		}
        // add an arc between two variables (there is direction)
		public void addEdge(Integer node1, Integer node2) {
			LinkedHashSet<Integer> adjacent = map.get(node1);
			if(adjacent==null) {
				adjacent = new LinkedHashSet();
				map.put(node1, adjacent);
			}
			adjacent.add(node2);
		}
        // specify the two variables are adjacent (ignoring the arc direction)
		public void addAdjacency(Integer node1, Integer node2) {
	        addEdge(node1, node2);
	        addEdge(node2, node1);
	    }
        
		public LinkedList<Integer> adjacentNodes(Integer last) {
			LinkedHashSet<Integer> adjacent = map.get(last);
			if(adjacent==null) {
				return new LinkedList();
			}
			return new LinkedList<Integer>(adjacent);
		}

	}

	// *********************************************************************



	public static void main(String[] args) {
		// this graph is directional
		int[][] matrix = new int[4][4];

		for (int n = 0; n < 4; n++)
		{
			for (int m = 0; m < 4; m++)
			{
				matrix[n][m] = 0;
			}
		}
		
		//matrix[0][1] = 1;
		matrix[0][2] = 1;
		matrix[1][2] = 1;
		matrix[2][3] = 1;
		matrix[1][3] = 1;
		// GetAllPaths(int[][] currentMatrix, int currentVarNum, int startVar, int endVar, boolean flag)
		// flag: whether to care about the arc direction
		GetAllPaths gp = new GetAllPaths(matrix, 4, 1, 4, false);
		ArrayList<ArrayList<Integer>> paths = gp.getResult();
		
		System.out.println("Size: " + paths.size());
		
		for(ArrayList<Integer> path: paths)
		{
			System.out.println("current size:" + path.size());
			for(Integer node: path)
			{
				System.out.print(node);
			}
			
			System.out.println();
		}

	}

}


