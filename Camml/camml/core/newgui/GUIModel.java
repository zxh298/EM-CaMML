/*
 *  [The "BSD license"]
 *  Copyright (c) 2002-2011, Rodney O'Donnell, Lloyd Allison, Kevin Korb
 *  Copyright (c) 2002-2011, Monash University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *    3. The name of the author may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.*
 *
 *  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 *  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 *  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 *  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package camml.core.newgui;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.math3.random.MersenneTwister;

import camml.core.latentDetect.LatentDetect;
import camml.core.library.WallaceRandom;
import camml.core.models.ModelLearner;
import camml.core.models.bNet.BNet;
import camml.core.search.EM;
import camml.core.search.MMLEC;
import camml.core.search.MetropolisSearch;
import camml.core.search.Node;
import camml.core.search.SearchPackage;
import camml.core.search.TOM;
import camml.core.search.TOMCoster.UniformTOMCoster;
import camml.core.searchDBN.DTOM;
import camml.core.searchDBN.ExportDBNNetica;
import camml.core.searchDBN.MetropolisSearchDBN;

import camml.plugin.friedman.FriedmanWrapper;
import camml.plugin.netica.NeticaFn;
import camml.plugin.rodoCamml.RodoCammlIO;
import camml.plugin.tomCoster.ExpertElicitedTOMCoster;
import camml.plugin.weka.Converter;
import cdms.core.Type;
import cdms.core.Value;
import cdms.core.Value.Vector;
import cdms.core.VectorFN;
import cdms.core.VectorFN.WeightedVector;

/**
 * Code for actually running a search, exporting results etc. Designed to work
 * closely with the cammLGUI class: i.e. a GUIModel object is passed to the
 * cammlGUI constructor.<br>
 * 
 * Much of this class amounts to a wrapper for existing CaMML methods, as well
 * as checks on the parameters selected by the user via the GUI. <br>
 * See also:<br>
 * - GUIParameters<br>
 * - CaMMLGUI
 * 
 * The discovering and learning with latent variable part is done by integrating
 * MCMC with EM algorithm.
 *
 * @author Alex Black and Xuhui Zhang
 */
public class GUIModel implements GUIParameters {
	protected MetropolisSearch metropolisSearch = null;

	// The input fully observed data:
	Value.Vector data = null;
	// the number of observed nodes
	private int numNodes;
	// learn with latent or not
//	public static boolean searchLatent = false;
	public static boolean searchLatent = true;
	
	private String[][] allObservedNodesStates;
	private List<String> observedVarStateCombinations = new ArrayList<String>();
	// Search parameters:
	protected ModelLearner MMLLearner = null;
	protected String LatentInitialisation = null;
	protected double searchFactor = 1.0;
	protected int maxSECs = 30;
	protected double minTotalPosterior = 0.999;

	protected File selectedFile = null; // Used mainly for path - for file dialog boxes etc
	protected File lastExportedBNet = null;

	protected Random r = RNGs[0]; // Random number generator. Defined in GUIParameters interface
	protected boolean useSetSeed = RNGUseSetSeed[0];
	protected boolean useSetSeed2 = RNGUseSetSeed2[0];
	// Note: random seeds for WallaceRandom must be integers, seed for
	// java.util.Random is long (hence integer OK)
	protected int randomSeed = new Random().nextInt(); // Updated based on what user enters in GUI...
	protected int randomSeed2 = new Random().nextInt(); // Wallace Random needs 2 parameters for seed...

	// the alpha value used in the conditional statistical test
	protected double alpha = 0.05;
	protected double errorRate = 0.005;
	protected int EMIteration = 10;
	protected double EMThreshold = 0.000001;
	// whether latent variable is detected
	protected int latentArity = 2;
	int result_node_number;
	double finalLatentModelCost = Double.POSITIVE_INFINITY;

	// Expert Priors
	protected boolean useExpertPriors = false;
	// String of the expert priors entered by the user
	String expertPriorsString = null;

	// Results for standard BNs
	protected Value.Vector searchResults = null;
	// Results for learning DBNs
	protected MMLEC[] searchResultsDBN = null;
	// Results for only observed nodes in latent model searching.
	protected Value.Vector searchResults_observed = null;
	
	/**
	 * Export ONE network (Netica format) from the set of 'representative networks'
	 * in the full results to a specified location. (i.e. after combining TOMs ->
	 * SECs -> MMLECs )
	 * 
	 * @param filepath
	 *            Path and file name to save the specified network.
	 * @param index
	 *            Index of network in full results array. NOTE: index appended to
	 *            filename automatically.
	 * @throws Exception
	 *             IO errors, invalid index etc
	 */
	public void exportFullResultsBN(String filepath, int index) throws Exception {
		if (metropolisSearch == null || !metropolisSearch.isFinished())
			throw new Exception("Cannot produce network if search has not been run.");
		if (searchResults == null && searchResultsDBN == null)
			throw new Exception("Search results: Not generated.");
		if (index < 0)
			throw new Exception("Invalid Index.");
		if (searchResults != null && index > searchResults.length() - 1)
			throw new Exception("Invalid Index.");
		if (searchResultsDBN != null && index > searchResultsDBN.length - 1)
			throw new Exception("Invalid Index.");

		System.out.println("index: " + index);
		
		// Generate the file path:
		String path;
		if (filepath.endsWith(".dne")) {
			path = filepath.substring(0, filepath.length() - 4);
		} else {
			path = filepath;
		}

		path = path + index + ".dne"; // Concatenate number at end

		if (searchResults != null) { // Export standard BN
			Value.Structured repNetwork = (Value.Structured) MMLEC.getRepresentative.apply(searchResults.elt(index));

			// File path, Network, parameters
			Value.Structured saveStruct = new Value.DefStructured(new Value[] { new Value.Str(path),
					(BNet) repNetwork.cmpnt(0), (Value.Vector) repNetwork.cmpnt(1) });

			try {
				NeticaFn.SaveNet SaveNet = new NeticaFn.SaveNet();
				SaveNet.apply(saveStruct);
			} catch (Exception e) {
				System.out.println("Error: Saving netica BN failed!");
			}

		} else { // Export DBN
			// Get the representative network:
			DTOM repNetwork = (DTOM) searchResultsDBN[index].getSEC(0).getTOM(0);
			ExportDBNNetica.export(path, repNetwork, "_0", "_1");
		}

	}

	/**
	 * Exports ALL networks (Netica format) from the set of 'representative
	 * networks' in the full results to a specified location. NOTE: Number of
	 * network (sorted by posterior) appended to filename.
	 * 
	 * @param filepath
	 *            Path and filename to export networks to.
	 * @throws Exception
	 *             On IO errors, etc
	 */
	public void outputFullResultsAllBNs(String filepath) throws Exception {
		if (metropolisSearch == null || !metropolisSearch.isFinished())
			throw new Exception("Cannot produce network if search has not been run.");
		if (searchResults == null && searchResultsDBN == null)
			throw new Exception("Search results: Not generated.");

		int numBNs;
		if (searchResults != null)
			numBNs = searchResults.length();
		else
			numBNs = searchResultsDBN.length;

		for (int i = 0; i < numBNs; i++) {
			exportFullResultsBN(filepath, i);
		}
	}

	/** Load data file. See loadDataFile( String path ) */
	public void loadDataFile(File f) throws Exception {
		loadDataFile(f.getAbsolutePath());
	}

	/**
	 * Load data from a file, using absolute path. Currently supported formats:
	 * arff, cas and data
	 * 
	 * @param path
	 *            Location of data file (inc. file name)
	 * @throws Exception
	 *             If file not found, error loading file, etc
	 */
	public void loadDataFile(String path) throws Exception {
		if (path.endsWith("arff")) {
			try {
				// TODO: Currently set NOT to discretize continuous or replace missing values.
				data = Converter.load(path, false, false);
				numNodes = ((Value.Structured) data.elt(0)).length();
				// System.out.println("File loaded OK");
			} catch (FileNotFoundException e) {
				data = null;
				throw new Exception("File not found. " + e);
			} catch (IOException e) {
				data = null;
				throw new Exception("IO Exception: " + e);
			} catch (Exception e) {
				data = null;
				throw new Exception("Error loading file: " + e);
			}
			return;
		}

		if (path.endsWith("cas")) {
			try {
				data = RodoCammlIO.load(path);
				numNodes = ((Value.Structured) data.elt(0)).length();
			} catch (FileNotFoundException e) {
				data = null;
				throw new Exception("File not found. " + e);
			} catch (IOException e) {
				data = null;
				throw new Exception("IO Exception: " + e);
			} catch (Exception e) {
				data = null;
				throw new Exception("Error loading file: " + e);
			}
			return;
		}

		// TODO: Currently untested. FriedmanWrapper.loadData(...) seems to require a
		// ".names" file in addition to the ".data" file???
		if (path.endsWith("data")) {
			try {
				data = FriedmanWrapper.loadData(path);
				numNodes = ((Value.Structured) data.elt(0)).length();
			} catch (FileNotFoundException e) {
				data = null;
				throw new Exception("File not found. " + e);
			} catch (IOException e) {
				data = null;
				throw new Exception("IO Exception: " + e);
			} catch (Exception e) {
				data = null;
				throw new Exception("Error loading file: " + e);
			}
			return;
		}

		// If file extension not matched by now: Unknown format.
		data = null;
		throw new Exception("Unknown file format.");
	}

	/**
	 * Checks a dataset to determine if discrete or continuous. Returns true if
	 * discrete (not continuous); false otherwise.
	 */
	// TODO: Weka plugin does have the ability to deal with missing and continuous
	// data: we might want this to be implemented/tested!
	public boolean checkDataNotContinuous() {
		if (data == null)
			return true;

		// First: determine number of variables:
		int numVars = ((Type.Structured) ((Type.Vector) data.t).elt).labels.length;

		for (int i = 0; i < numVars; i++) {
			if (data.cmpnt(i).elt(0).t instanceof cdms.core.Type.Continuous) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Validates format of expert priors, entered as a string. Note: Validity of a
	 * string of priors depends on data (for variable names etc)
	 * 
	 * @param ExpertPriors
	 *            String encoding the expert priors
	 * @return True if format acceptable, false if it has errors
	 * @throws Exception
	 *             If format invalid (exception text has reason + line number why)
	 */
	public boolean validateExpertPriors(String ExpertPriors) throws Exception {
		StringReader rd = new StringReader(ExpertPriors);
		try {
			new ExpertElicitedTOMCoster(0.5, rd, data);
		} catch (RuntimeException e) {
			// Constructor threw exception -> error in Expert Prior text...
			throw new Exception(e);
		}
		return true;
	}

	/**
	 * Checks if the loaded data is valid. More checks in this function would
	 * probably be a good idea...
	 */
	public boolean dataValid() {
		// TODO: More checks? i.e. only discrete or symbolic, not continuous...
		return data != null;
	}

	public boolean MMLLearnerValid() {
		return MMLLearner != null;
	}

	/**
	 * Method to check if expert priors are OK.
	 * 
	 * @return True if expert priors OK, or expert priors are not used.
	 */
	public boolean expertPriorsValid() {
		if (useExpertPriors) {
			if (expertPriorsString == null)
				return false;
			try {
				validateExpertPriors(expertPriorsString);
			} catch (Exception e) {
				return false;
			}
		}

		return true;
	}

	public boolean searchFactorValid() {
		return (searchFactor >= GUIParameters.minSearchFactor && searchFactor <= GUIParameters.maxSearchFactor);
	}

	public boolean maxSECsValid() {
		return (maxSECs >= GUIParameters.minSECs && maxSECs <= GUIParameters.maxSECs);
	}

	public boolean minTotalPosteriorValid() {
		return (minTotalPosterior >= GUIParameters.min_minTotalPosterior
				&& minTotalPosterior <= GUIParameters.max_minTotalPosterior);
	}

	public boolean alphaValid() {
		return (alpha >= GUIParameters.minAlpha && alpha <= GUIParameters.maxAlpha);
	}

	public boolean errorRateValid() {
		System.out.println();
		return (errorRate >= GUIParameters.minErrorRate && errorRate <= GUIParameters.maxErrorRate);
	}

	public boolean EMIterationValid() {
		return (EMIteration >= GUIParameters.minEMIteration && EMIteration <= GUIParameters.maxEMIteration);
	}

	public boolean EMThresholdValid() {
		return (EMThreshold >= GUIParameters.minEMThreshold && EMThreshold <= GUIParameters.maxEMThreshold);
	}

	public boolean latentArityValid() {
		return (latentArity >= GUIParameters.minLatentArity && latentArity <= GUIParameters.maxLatentArity);
	}

	/**
	 * Method that actually runs the full Metropolis Search. Does not check if
	 * data/parameters are valid: assumes these checks have already been conducted.
	 * Once search complete: results can be obtained from the appropriate GUIModel
	 * object: i.e.<br>
	 * - GUIModel.searchResults, or<br>
	 * - GUIModel.searchResultsDBN
	 * 
	 * @param learnDBN
	 *            False if learning a standard BN; True if learning a DBN
	 */
	public void runSearch(boolean learnDBN) {

		if (numNodes < 2) {
			System.out.println(" The node number should not be less than two ! ");
			return;
		}

		// Reset search results in case already run before this session:
		searchResults = null;
		searchResultsDBN = null;

		if (!learnDBN) { // Learn a normal BN
			// Create MetropolisSearch object:
			metropolisSearch = new MetropolisSearch(r, data, MLLearner, MMLLearner);
			// Set expert priors, if they are being used:
			if (useExpertPriors) {
				Value.Str s = new Value.Str(expertPriorsString);
				metropolisSearch.setOption("TOMPrior", s);
			}
			System.out.println(" -- Learning Bayesian Network -- ");
		} else { // Use DBN code to learn a DBN
			// Create a MetropolisSearchDBN object:
			metropolisSearch = new MetropolisSearchDBN(r, data, MLLearner, MMLLearner);
			System.out.println(" -- Learning Dynamic Bayesian Network (DBN) -- ");
		}

		// Change settings:
		metropolisSearch.caseInfo.searchFactor = searchFactor; // No separate value kept in MetropolisSearch
		metropolisSearch.caseInfo.maxNumSECs = maxSECs; // No separate value kept in MetropolisSearch
		metropolisSearch.caseInfo.minTotalPosterior = minTotalPosterior; // No separate value kept in MetropolisSearch

		// Display whether using set RNG seed or random, plus what the seed is:
		System.out.print("RNG = " + r);
		if (useSetSeed) {
			if (r.getClass() == Random.class) { // Java random
				System.out.println("; Set seed = " + randomSeed);
				r.setSeed(randomSeed);
			} else if (r.getClass() == WallaceRandom.class) {
				System.out.println("; Set seeds = " + randomSeed + " & " + randomSeed2);
				((WallaceRandom) r).setSeed(new int[] { randomSeed, randomSeed2 });
			}
		} else {
			System.out.println("; Random seed");
		}

		// Run the MetropolisSearch algorithm until finished:
		int count = 0;
		while (!metropolisSearch.isFinished()) {
			metropolisSearch.doEpoch();
			count++;
		}
		System.out.println("\n\nSearch finished... " + count + " loops");

		System.out.println("\n\n===================================");
		System.out.println("----- Generating Results -----");
		generateFullResults();

	}

	/**
	 * Method that actually runs the full Metropolis Search with latent variable.
	 * Does not check if data/parameters are valid: assumes these checks have
	 * already been conducted. Once search complete: results can be obtained from
	 * the appropriate GUIModel object: i.e.<br>
	 * - GUIModel.searchResults, or<br>
	 * - GUIModel.searchResultsDBN
	 * 
	 * The current version of CaMML does not support learning latent variable in
	 * DBN.
	 */

	public void runSearchLatent() throws Exception {

		// as there is no trigger contains less than four variables, so stop.
		if (numNodes < 4) {
			System.out.println(" The node number should not be less than four ! ");
			return;
		}

		// as if the number of node is more than seven, there could be more than one
		// triggers
		if (numNodes > 7) {
			System.out.println(" The node number should not be more than seven ! ");
			return;
		}

		System.out.println("Detecting latent varaible...");
		LatentDetect ld = new LatentDetect(data, alpha, errorRate);
		ld.run();
		
		boolean latentDetected = ld.getMatch();

		/**
		 * If a latent variable is detected, we will construct a matrix based on the
		 * matched subset of variables (including the latent). Then we randomly (could
		 * be either parents or children) connect the rest (remained) ones to the subnet
		 * in order to build the initial structure for running EM.
		 */
			
		// if there is one latent variable detected:
		if (latentDetected) {
					
			System.out.println();
			System.out.println("Trigger matched!");

			 // get the matched trigger structure
			 int[][] hidden_model = ld.getMatchedTriggerStructure();
			 // get the variable index that matches trigger
			 int[] matchedIndex = ld.getMatchedSubsetIndeces();
			 int[][] latent_matrix = getLatentMatrix(hidden_model, matchedIndex);
					
			/**
			 * As the structure of the trigger subnet is fixed, we use CaMML to find the
			 * best structure of how to connect the remained variables.
			 */
	
			// generate random expected counts to fill up the data of the latent variable
			Value.Vector data_filled = makeEMFakeData(data, latentArity);
			Type.Structured eltType = ((Type.Structured) (((Type.Vector) data_filled.t).elt));
			String headers_filled[] = eltType.labels;
			Value.Str s = getExpertPriorStr(latent_matrix, headers_filled, matchedIndex);

			/** EM starts*/
			EM em = new EM(latent_matrix, data_filled, EMIteration, latentArity, EMThreshold);

			double oldMMLScore = Double.POSITIVE_INFINITY;
			em.Initialise();
			for (int i = 0; i < EMIteration; i++) {

				searchResults = null;

				if (i == 0) {
					// cuz data_filled already contains the initial expected counts, so no need to
					// run E-Step in the first EM iteration.
					metropolisSearch = new MetropolisSearch(r, data_filled, MLLearner, MMLLearner);

				} else {
					// get current expected counts using E-step
					em.EStep();

					metropolisSearch = new MetropolisSearch(r, em.getWeightedEMFakeData(), MLLearner, MMLLearner);
				}

				metropolisSearch.setOption("TOMPrior", s);
				metropolisSearch.caseInfo.searchFactor = searchFactor;
				metropolisSearch.caseInfo.maxNumSECs = maxSECs;
				metropolisSearch.caseInfo.minTotalPosterior = minTotalPosterior;

				// Run the MetropolisSearch algorithm until finished:
				int count = 0;
				while (!metropolisSearch.isFinished()) {
					metropolisSearch.doEpoch();
					count++;
				}
				
				// double newMLScore = em.getMLScore();
//				double newMLScore = camml_mlcost;
//				double oldPlusNewScore = Math.abs(oldMLScore) + Math.abs(newMLScore);
//				double avgMLScore = (oldPlusNewScore + Math.ulp(oldPlusNewScore)) / 2;
//				double deltaScore = Math.abs(newMLScore - oldMLScore);
				double newMMLScore = metropolisSearch.getBestCost();


//				if ((deltaScore / avgMLScore) < EMThreshold) {
				if (Math.abs(newMMLScore - oldMMLScore) < Math.abs(oldMMLScore) * 0.001 || i == EMIteration-1) {	
					generateLatentFullResults();
					
					// terminate the search
					break;

				} else {
					System.out.println();
					System.out.println("EM-CaMML iteration: " + i);
					System.out.println("MML cost: " + metropolisSearch.getBestCost());
					
					oldMMLScore = newMMLScore;
					
					em.updateStructure(metropolisSearch.getBestTOM());

					// re-estimate cpt probabilities using M-Step
					em.MStep();
					
					// free up memory immediately after each EM iteration.
					metropolisSearch = null;
				}
			}
			
			System.out.println();
			System.out.println("Search finished. Please see the results.");
		}

		// if no latent is detected:
		if (!latentDetected) {
			/**
			 * For the existing of the latent node (if not detected), we need to compare the
			 * learned latent model with the best learned model which the latent is
			 * disconnected.
			 * 
			 * In order to do a fair comparison, we use the final expected counts for
			 * learning the latent model, to learn a model which the latent is disconnected.
			 * We achieve this by coping the expected counts, and use expert prior to
			 * disconnect the latent node.
			 * 
			 */

			System.out.println();
			System.out.println("No trigger matched!");
			System.out.println("Now add a random latent variable in the search...");
			/**
			 * STEP ONE: Since we choose random parameters (expected counts), if we
			 * immediately run a search, the latent variable will end up disconnected
			 * (independent) from the rest of the network in most cases. So we run the 
			 * standard EM for several times on a fixed initial network, thus some dependencies 
			 * (shown in the result expected counts) could happen between the latent and 
			 * the observed variables.
			 * 
			 * There are three ways to initialize the fixed initial network to run the standard EM:
			 * 
			 * 1) Set the latent node as parent of every fully observed node. This Naive-Bayes 
			 *    structure requires strong prior knowledge. (see function "getInitialStructure1").
			 *    
			 * 2) Another option is to check marginal dependency between every two variables,
			 *    if two variables are marginal dependent to each other, then both will be the
			 *    parents of the latent node in the initial structure. Otherwise will be the
			 *    children of the latent. For example, if the marginal dependency matrix is:
			 *    
			 *    0 0 1 1 
			 *    0 0 1 1
			 *    1 1 0 1 
			 *    1 1 1 0
			 *    
			 *    Then the first two nodes will be parents and the last two will be children.
			 *    (see function "getInitialStructure2")
			 * 
			 * 3) Start by a random structure which the latent node is connected
			 */		
			
			int[][] initial_latent_matrix = getInitialStructure1();
			
			if(LatentInitialisation.equals("latent as root"))
			{
				initial_latent_matrix = getInitialStructure1();
			}
			if(LatentInitialisation.equals("using dependencies"))
			{
				initial_latent_matrix = getInitialStructure2(ld.getMarginalDependencyMatrix());
			}
			if(LatentInitialisation.equals("random"))
			{
				Value.Vector data_filled = makeEMFakeData(data, latentArity);
				initial_latent_matrix = getInitialStructure3(data_filled);
			}
			
			// generate random expected counts to fill up the data of the latent variable
			Value.Vector data_filled = makeEMFakeData(data, latentArity);

			/** Run standard EM first to get good expected counts */
			System.out.println("Running standard EM to get good expected counts...");
			EM em_step1 = new EM(initial_latent_matrix, data_filled, EMIteration, latentArity, EMThreshold);
			em_step1.Run();
			WeightedVector data_step1 = em_step1.getWeightedEMFakeData();
			System.out.println();
			System.out.println("Running standard EM finished.");

			/**
			 * STEP TWO: get the expected counts from step one, and run EM-CAMML to find a
			 * good latent model. It updates structure and expected during the EM algorithm.
			 * 
			 */

			WeightedVector data_step2 = data_step1;

			EM em_step2 = new EM(initial_latent_matrix, data_step2, EMIteration, latentArity, EMThreshold);

			// initialise Maximum likelihood score:
			double step2_oldMMLScore = Double.POSITIVE_INFINITY;

			em_step2.Initialise();

			for (int i = 0; i < 30; i++) {

				searchResults = null;

				// use the expected counts got from standard EM, no need to run E Step to
				// estimate
				if (i == 0) {
					metropolisSearch = new MetropolisSearch(r, data_step2, MLLearner, MMLLearner);
				}

				else {
					// current expected counts based on current cpt parameters
					em_step2.EStep();
					try {
						metropolisSearch = new MetropolisSearch(r, em_step2.getWeightedEMFakeData(), MLLearner, MMLLearner);
					}
					catch(Exception e){
						System.out.println("Error");
						Value.Vector data1 = em_step2.getWeightedEMFakeData();
						System.out.println("Error");
					}
				}

				metropolisSearch.caseInfo.searchFactor = searchFactor;
				metropolisSearch.caseInfo.maxNumSECs = maxSECs;
				metropolisSearch.caseInfo.minTotalPosterior = minTotalPosterior;

				// Run the MetropolisSearch algorithm until finished:
				int count = 0;
				while (!metropolisSearch.isFinished()) {
					metropolisSearch.doEpoch();
					count++;
				}

				double step2_newMMLScore = metropolisSearch.getBestCost();
				
				// double newMLScore = em_step2.getMLScore();
//				double newMLScore = camml_mlcost;
//
//				double oldPlusNewScore = Math.abs(step2_oldMLScore) + Math.abs(newMLScore);
//				double avgMLScore = (oldPlusNewScore + Math.ulp(oldPlusNewScore)) / 2;
//				double deltaScore = Math.abs(newMLScore - step2_oldMLScore);

				if (Math.abs(step2_newMMLScore - step2_oldMMLScore) < Math.abs(step2_oldMMLScore) * 0.001 ||  i == EMIteration-1) {

					/**
					 * MML score Adaptive code has three parts:
					 * 
					 * msglen(structure) + msglen(parameters) + msglen(data|structure+parameters)
					 * 
					 * the second and third parts are calculated node by node.
					 * 
					 * If a model has a latent node, when compare with a fully observed node, we
					 * dont include the third part. Any other nodes including children of the
					 * latent, we score them normally.
					 * 
					 */
					TOM current_bestTOM = metropolisSearch.getBestTom();

//					double oldLatentNodeCost = current_bestTOM.getNode(0).cost(MMLLearner,
//							em_step2.getWeightedEMFakeData());

					/** cost without MML correction (only the third part of the original score) */
					double newLatentNodeCost = current_bestTOM.getNode(0).cost(SearchPackage.LatentCPTLearner,
							em_step2.getWeightedEMFakeData());

					finalLatentModelCost = metropolisSearch.getBestCost() - newLatentNodeCost;

					searchResults = metropolisSearch.getResults();

					result_node_number = metropolisSearch.getBestTom().getNumNodes();
					
//					// save the best tom to as Netica dne file to local:
//					Value.Structured result_tom_Network = metropolisSearch.getBestStruct(MMLLearner);
//					exportBestTom(result_tom_Network, exportPath);
					
					break;

				} else {
					step2_oldMMLScore = step2_newMMLScore;
					// update the structure of EM based on the result of metropolisSearch
					em_step2.updateStructure(metropolisSearch.getBestTOM());
					
					// run EM M Step to re-estimate cpt probabilities
					em_step2.MStep();
					
					System.out.println();
					System.out.println("EM-CaMML iteration: " + i);
					System.out.println("ML score: " + step2_newMMLScore);
					System.out.println("MML cost: " + metropolisSearch.getBestCost());
					
					// free up memory immediately after each EM iteration.
					metropolisSearch = null;
				}
			}
			

			/**
			 * STEP THREE: Compare the learned latent model with best learned fully observed model. 
			 */
			System.out.println();
			System.out.println("Now verifying the latent model with fully observed model..");

			
			double bestLatentCost1 = finalLatentModelCost;
			double bestObservedCost2 = getMMLCost_onlyObserved(data);
			
			
			// ignore the latent model
			if (bestObservedCost2 <= bestLatentCost1) {
				System.out.println("\nThe latent variable is not necessary to be in the structure.");
				System.out.println("Now run CaMML without latent node...");
				System.out.println();
				System.out.println();

				// run the normal search without considering latent node.
				runSearch(false);

			} else {
				System.out.println("\nA good latent model found. Please see the result.");
			}
		}
	}

	/**
	 * Generate full results. After calling this function, results can be obtained
	 * from: GUIModel.searchResults or GUIModel.searchResultsDBN
	 * 
	 * @return True if results generated successfully, false otherwise (i.e. search
	 *         not run)
	 */
	public boolean generateFullResults() {
		if (metropolisSearch == null)
			return false;
		if (!metropolisSearch.isFinished())
			return false;

		if (metropolisSearch instanceof MetropolisSearchDBN) { // Learning a DBN
			searchResultsDBN = ((MetropolisSearchDBN) metropolisSearch).getResultsMMLEC();
		} else { // Learning a standard BN (with or without latent variable)
			searchResults = metropolisSearch.getResults();
		}
		return true;
	}

	public boolean generateLatentFullResults() {
		if (metropolisSearch == null)
			return false;
		if (!metropolisSearch.isFinished())
			return false;

		// Learning a standard BN (with or without latent variable)
		searchResults = metropolisSearch.getResults();

		return true;
	}

	/**
	 * Return one of the representative networks (from full results) as a String
	 * String intended to be used by BNetViewer class to display a network.
	 * 
	 * @param index
	 *            Index of the representative network to be returned as a string.
	 * @return String of network.
	 * @throws Exception
	 *             If search not run, full results not generated, or bad index
	 *             passed.
	 */
	public String generateNetworkStringFullResults(int index) throws Exception {
		if (metropolisSearch == null || !metropolisSearch.isFinished())
			throw new Exception("Cannot produce network if search has not been run.");
		if (searchResults == null && searchResultsDBN == null)
			throw new Exception("Error: Search results not generated.");
		if (index < 0)
			throw new Exception("Invalid Index.");
		if (searchResults != null && index > searchResults.length() - 1)
			throw new Exception("Invalid Index.");
		if (searchResultsDBN != null && index > searchResultsDBN.length - 1)
			throw new Exception("Invalid Index.");

		if (searchResults != null) { // Standard BN
			// (Model,params)
			Value.Structured repNetwork = (Value.Structured) MMLEC.getRepresentative.apply(searchResults.elt(index));

			BNet network = (BNet) repNetwork.cmpnt(0);

			String filename = selectedFile.getName();
			if (filename == null || filename.equals(""))
				filename = "Network";
			filename = filename + " - " + index;

			return network.export(filename, (Value.Vector) repNetwork.cmpnt(1), "netica");

		} else { // DBN
			MMLEC m = searchResultsDBN[index];

			DTOM repNetwork = (DTOM) m.getSEC(0).getTOM(0);

			return ExportDBNNetica.makeNeticaFileString(repNetwork, "_0", "_1");
		}
	}

	/**
	 * this function is borrowed from EM.java which makes fake data based on
	 * original data for running EM. Please go to EM.java to see more details.
	 * 
	 * Author: Xuhui Zhang
	 * 
	 * @throws IOException
	 * @throws NumberFormatException
	 */
	private WeightedVector makeEMFakeData(Value.Vector data, int latentArity)
			throws NumberFormatException, IOException {

		allObservedNodesStates = new String[numNodes][];

		String[] latentStates = new String[latentArity];
		for (int i = 0; i < latentArity; i++) {
			 latentStates[i] = "H" + i;
//			latentStates[i] = String.valueOf(i);
		}

		// get the headers of the original data:
		Type.Structured eltType = ((Type.Structured) (((Type.Vector) data.t).elt));

		String headers[] = eltType.labels;
		// set up the headers with the name of the latent variable for the fake data
		String[] headersWithLatent = new String[headers.length + 1];
		headersWithLatent[0] = "H";

		for (int i = 0; i < headers.length; i++) {
			headersWithLatent[i + 1] = headers[i];
		}

		// the fake data include the latent fake values as well as the original data;
		String[][] data_fake = new String[headersWithLatent.length][data.length() * latentArity];
		TreeMap<String, Integer>[] columnValues = (TreeMap<String, Integer>[]) new TreeMap[headersWithLatent.length];

		for (int c = 0; c < headersWithLatent.length; c++) {
			columnValues[c] = new TreeMap<String, Integer>();
		}

		// put all the observed variable states in
		for (int i = 0; i < eltType.cmpnts.length; i++) {
			Type d = eltType.cmpnts[i];
			String[] names = (String[]) ((Type.Symbolic) d).ids;
			for (int n = 0; n < names.length; n++) {
				columnValues[i + 1].put(names[n], 1);
			}

			allObservedNodesStates[i] = names;
		}

		// get all possible combinations of all node states
		recursive_vector(0, "", allObservedNodesStates);

		// the fake data only for the latent variable;
		String[] latent_fakedata = new String[data.length() * latentArity];

		for (int i = 0; i < latentStates.length; i++) {
			for (int n = 0; n < data.length(); n++) {
				latent_fakedata[data.length() * i + n] = latentStates[i];
			}
		}

		for (int n = 0; n < data.length() * latentArity; n++) {
			data_fake[0][n] = latent_fakedata[n];
			columnValues[0].put(data_fake[0][n], 1);
			for (int m = 0; m < headers.length; m++) {
				data_fake[m + 1][n] = data.cmpnt(m).elt(n % data.length()).toString();
			}
		}

		Value.Vector[] vecArray = new Value.Vector[headersWithLatent.length];

		for (int c = 0; c < headersWithLatent.length; c++) {
			Set<String> keysPre = columnValues[c].keySet();
			String[] keys = new String[keysPre.size()];
			keysPre.toArray(keys);

			for (int i = 0; i < keys.length; i++) {
				columnValues[c].put(keys[i], i);
			}

			Type.Symbolic type = new Type.Symbolic(false, false, false, false, keys);

			int[] intArray = new int[data_fake[c].length];
			for (int j = 0; j < intArray.length; j++) {
				String v = data_fake[c][j];
				intArray[j] = columnValues[c].get(v);
			}

			// combine type and value together to form a vector of symbolic values.
			vecArray[c] = new VectorFN.FastDiscreteVector(intArray, type);
		}

		Value.Structured vecStruct = new Value.DefStructured(vecArray, headersWithLatent);
		Value.Vector vec = new VectorFN.MultiCol(vecStruct);

		int total_length = data.length() * latentArity;

		// generate random weights and add in the fake data
		double[] weights = new double[total_length];

		ArrayList<double[]> weight_list = new ArrayList<double[]>();

		for (int i = 0; i < observedVarStateCombinations.size(); i++) {
			double[] numbers = generateRandomWeights(latentArity);
			weight_list.add(numbers);
		}

		ArrayList<String[]> state_combinations = new ArrayList<String[]>();
		for (String s : observedVarStateCombinations) {
			String[] state_comb = s.split(",");
			state_combinations.add(state_comb);
		}

		for (int i = 0; i < data.length(); i++) {
			String[] rowdata = new String[numNodes];
			for (int n = 0; n < numNodes; n++) {
				rowdata[n] = data.cmpnt(n).elt(i).toString();
			}

			// the same observed data row with the same latent node state should have the same expected count 
			for (int n = 0; n < state_combinations.size(); n++) {
				if (Arrays.equals(rowdata, state_combinations.get(n))) {
					for (int m = 0; m < latentArity; m++) {
						double[] values = weight_list.get(n);
						weights[m * data.length() + i] = values[m];
					}
				}
			}
		}

		WeightedVector weightedEMfakedata = new WeightedVector(vec, weights);

		return weightedEMfakedata;
	}

	/**
	 * these random weights depend on how many states of the latent variable.
	 * 
	 * @throws IOException
	 */
	private double[] generateRandomWeights(int stateNum) throws IOException {
		// apply the Mersenne Twister random number generator:
		MersenneTwister rand = new MersenneTwister();
		// Random rand = new Random();

		double[] parameters = new double[stateNum];
		double sum = 0.0;

		for (int i = 0; i < parameters.length; i++) {
			parameters[i] = rand.nextDouble();
			sum += parameters[i];
		}

		for (int i = 0; i < parameters.length; i++) {
			parameters[i] /= sum;
		}

		return parameters;
	}

	private static int randInt(int min, int max) {
		Random rand = new Random();
		int randomNum = rand.nextInt((max - min) + 1) + min;
		return randomNum;
	}

	private int[][] getLatentMatrix(int[][] hiddenModel, int[] matchedIndex) {

		/**
		 * Because we have to find out which nodes can match a trigger, when the number
		 * of node is greater than four.
		 * 
		 */

		// initialise the structure matrix
		int[][] result_latent_matrix = new int[numNodes + 1][numNodes + 1];
		for (int n = 0; n < numNodes + 1; n++) {
			for (int m = 0; m < numNodes + 1; m++) {
				result_latent_matrix[n][m] = 0;
			}
		}

		for (int n = 0; n < hiddenModel.length; n++) {
			for (int m = 0; m < hiddenModel.length; m++) {
				// as the index 0 is always the latent node
				if (n == 0 && hiddenModel[n][m] == 1)
					result_latent_matrix[0][matchedIndex[m - 1] + 1] = 1;
				if (n != 0 && hiddenModel[n][m] == 1)
					result_latent_matrix[matchedIndex[n - 1] + 1][matchedIndex[m - 1] + 1] = 1;
			}
		}

		return result_latent_matrix;
	}

	// build expert prior string based on a given model which includes a latent node
	private Value.Str getExpertPriorStr(int[][] latent_matrix, String[] headers, int[] matchedIndex) {
		// get the expert prior string of the matched trigger
		String hidden_model_s = "set {n=" + String.valueOf(numNodes + 1) + ";}";
		hidden_model_s += "ed{} ";
		hidden_model_s += "tier{} ";
		hidden_model_s += "arcs {";

		int[] index_latent_included = new int[matchedIndex.length + 1];

		index_latent_included[0] = 0;

		for (int i = 0; i < matchedIndex.length; i++) {
			index_latent_included[i + 1] = matchedIndex[i] + 1;
		}

		// make matched trigger matrix expert string
		for (int n = 0; n < latent_matrix.length; n++) {
			for (int m = 0; m < latent_matrix[n].length; m++) {

				// we only build expert string among the matched trigger nodes,
				// and let CaMML to learn the structure among the rest.
				if (n != m) {

					boolean has_n = false;
					boolean has_m = false;
					for (int i = 0; i < index_latent_included.length; i++) {
						// because the first node is the latent
						if (index_latent_included[i] == n)
							has_n = true;
						if (index_latent_included[i] == m)
							has_m = true;
					}

					if (has_n && has_m) {
						hidden_model_s += headers[n];
						hidden_model_s += " ";
						hidden_model_s += "->";
						hidden_model_s += headers[m];
						hidden_model_s += " ";
						// there should be an arc if current cell is 1
						if (latent_matrix[n][m] == 1)
							hidden_model_s += "1.0; ";
						// there should not be an arc if current cell is 0
						else
							hidden_model_s += "0.0; ";

					}
				}

			}
		}
		hidden_model_s += "}";

		Value.Str s = new Value.Str(hidden_model_s);

		return s;
	}

	/** use CaMML MLLearner to get likelihood */
	private static double getCaMML_MLCost(TOM tom, ModelLearner mlModelLearner, Value.Vector data) {
		double result = 0.0;
		int numNodes = ((Value.Structured) data.elt(0)).length();
		for (int i = 0; i < numNodes; i++) {
			Node node = tom.getNode(i);
			double cost = node.cost(mlModelLearner, data);
			result += cost;
		}
		return result;
	}

	// initial a empty structure.
	private int[][] getInitialStructure() {
		int[][] latent_matrix = new int[numNodes + 1][numNodes + 1];
		for (int n = 0; n < numNodes + 1; n++) {
			for (int m = 0; m < numNodes + 1; m++) {
				latent_matrix[n][m] = 0;
			}
		}

		return latent_matrix;
	}

	// initial a structure than the latent is a parent of everything.
	private int[][] getInitialStructure1() {
		int[][] latent_matrix = new int[numNodes + 1][numNodes + 1];
		for (int n = 0; n < numNodes + 1; n++) {
			for (int m = 0; m < numNodes + 1; m++) {
				latent_matrix[n][m] = 0;
			}
		}
		for (int n = 1; n < numNodes + 1; n++)
			latent_matrix[0][n] = 1;

		return latent_matrix;
	}

	// initial a structure based on dependencies.
	private int[][] getInitialStructure2(int[][] marginalDependencyMatrix) {
		// start by add all observed nodes as children of the latent
		int[][] latent_matrix = getInitialStructure1();

        // if two observed variables are marginally independent, set them as two parents of the latent node

		for (int n = 0; n < marginalDependencyMatrix.length - 1; n++) {
			for (int m = n + 1; m < marginalDependencyMatrix.length; m++) {
				if (marginalDependencyMatrix[n][m] == 0) {
					// set nodes as parents of the latent
					latent_matrix[n + 1][0] = 1;
					latent_matrix[m + 1][0] = 1;

					latent_matrix[0][n + 1] = 0;
					latent_matrix[0][m + 1] = 0;
				}
			}
		}
		return latent_matrix;
	}

	// initial a random structure with some constraints 
    private int[][] getInitialStructure3(Value.Vector data) throws IOException{
    	
    	// the latent should not be a leaf (no children) node: A -> H
    	
    	// the latent should not be a parent (itself has no parent) has only one child: H->A
    	
    	// the latent should not be the middle node of a chain A -> H -> B
    	
    	// the latent should not be a disconnected node
    	 
//    	
    	 int nodeNum = ((Value.Structured) data.elt(0)).length();
    	 int[][] result = new int[nodeNum][nodeNum];
    	 
    	 Random rand = new Random();
    	 TOM t = new TOM(data);  
    	 
    	 // loop until we find a good model meets all constraints.
//    	 boolean correct = false;
    	 while(true)
    	 {
    		 t.randomOrder(rand);
        	 t.randomArcs(rand);
        	 
//        	 int nodeNum = t.getNumNodes();
        	 
        	 int[][] matrix = new int[nodeNum][nodeNum]; 
        	 
             for(int n=0; n<matrix.length; n++)
             {
            	 for(int m=0; m<matrix.length; m++)
            		 matrix[n][m] = 0;
             }
        	 
             for(int n=0; n<matrix.length; n++)
             {
            	 for(int m=0; m<matrix.length; m++)
            	 {
            		 if(t.isDirectedArc(n, m))
            			 matrix[n][m] = 1;
            	 }
             }
            
             int row_degree = 0;
             int column_degree = 0;
             
             for(int nn = 0; nn<matrix.length; nn++)
             {
            	 if(matrix[0][nn] == 1)
            		 row_degree++;
            	 if(matrix[nn][0] == 1)
            		 column_degree++;
             }
             
             if(row_degree > 1 || column_degree > 1)
             {
//            	 correct = true;
            	 
            	 result = matrix;
            	 
            	 break;
             }
             
    	 }
    	     	 
         return result;
    }
	
	// get the best MML score of searching among only fully observed models.
	private double getFullyObservedScore(Value.Vector data) {
		searchResults = null;
		metropolisSearch = new MetropolisSearch(r, data, MLLearner, MMLLearner);
		// Change settings:
		metropolisSearch.caseInfo.searchFactor = searchFactor; // No separate value kept in MetropolisSearch
		metropolisSearch.caseInfo.maxNumSECs = maxSECs; // No separate value kept in MetropolisSearch
		metropolisSearch.caseInfo.minTotalPosterior = minTotalPosterior; // No separate value kept in MetropolisSearch
		if (useSetSeed) {
			if (r.getClass() == Random.class) { // Java random
				r.setSeed(randomSeed);
			} else if (r.getClass() == WallaceRandom.class) {
				((WallaceRandom) r).setSeed(new int[] { randomSeed, randomSeed2 });
			}
		}

		int count = 0;
		while (!metropolisSearch.isFinished()) {
			metropolisSearch.doEpoch();
			count++;
		}

		double bestCost = metropolisSearch.getBestCost();
		searchResults_observed = metropolisSearch.getResults();
		searchResults = null;
		metropolisSearch = null;

		return bestCost;
	}

	private double getMMLCost_onlyObserved(Value.Vector data) {
		// run a search to get the best structure and score regarding to this structure
		MetropolisSearch metropolisSearch_ = new MetropolisSearch(r, data, MLLearner, SearchPackage.mmlCPTLearner);

		metropolisSearch_.caseInfo.searchFactor = searchFactor;
		metropolisSearch_.caseInfo.maxNumSECs = maxSECs;
		metropolisSearch_.caseInfo.minTotalPosterior = minTotalPosterior;

		while (!metropolisSearch_.isFinished()) {
			metropolisSearch_.doEpoch();
		}

		double bestScore = metropolisSearch_.getBestCost();

		return bestScore;
	}

	// get MML cost of a structure that the latent is disconnected by giving
	// expected counts.
	private double getMMLCost_latent_disconnected(Value.Vector data) {

		// build up the expert prior that the latent is disconnected.
		String model_s = "set {n=" + String.valueOf(numNodes + 1) + ";}";
		model_s += "ed{} ";
		model_s += "tier{} ";
		model_s += "arcs {";

		// disconnect the latent from every observed node by making the arc probability
		// 0:
		for (int i = 0; i < numNodes; i++) {
			model_s += "0 -> ";
			model_s += i + 1;
			model_s += " 0.0; ";

			model_s += i + 1;
			model_s += " -> 0 0.0; ";
		}

		model_s += "}";

		Value.Str s = new Value.Str(model_s);

		// run a search to get the best structure and score regarding to this structure
		MetropolisSearch metropolisSearch_ = new MetropolisSearch(r, data, MLLearner, SearchPackage.mmlCPTLearner);

		metropolisSearch_.setOption("TOMPrior", s);
		metropolisSearch_.caseInfo.searchFactor = searchFactor;
		metropolisSearch_.caseInfo.maxNumSECs = maxSECs;
		metropolisSearch_.caseInfo.minTotalPosterior = minTotalPosterior;

		while (!metropolisSearch_.isFinished()) {
			metropolisSearch_.doEpoch();
		}

		double bestScore = metropolisSearch_.getBestCost();

		TOM tom = metropolisSearch_.getBestTom();

		double old_latent_cost = tom.getNode(0).cost(SearchPackage.mmlCPTLearner, data);

		double new_latent_cost = tom.getNode(0).cost(MMLLearner, data);

		bestScore = bestScore - old_latent_cost + new_latent_cost;

		/**
		 * As there is extra cost by specifying structure (by guarantee the latent is
		 * disconnected). So in order to be fair when compare with the learned latent
		 * model, we subtract the structure cost (by using ExpertElicitedTOMCoster) from
		 * the total cost, and use the latent model TOM coster (UniformTOMCoster) to
		 * cost the structure and add the result to the total cost (by using the same
		 * arcProb learned from ExpertElicitedTOMCoster).
		 * 
		 */

		double structure_cost = metropolisSearch_.caseInfo.tomCoster.cost(tom);

		// temp += structure_cost;
		double learned_arcProb = metropolisSearch_.getArcProb();
		double new_structure_cost = UniformTOMCoster.cost(tom, learned_arcProb);

		// double temp_result = temp + new_structure_cost;
		double result = bestScore - structure_cost + new_structure_cost;

		return result;
	}

	// recursively generate all possible states combinations
	private void recursive_vector(int d, String str, String[][] states) {
		if (d == states.length) {
			observedVarStateCombinations.add(str);
			return;
		}
		for (int k = 0; k < states[d].length; k++) {
			recursive_vector(d + 1, str + states[d][k] + ",", states);
		}
		return;
	}


}
