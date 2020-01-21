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

import java.text.DecimalFormat;
import java.util.Random;

import camml.core.library.WallaceRandom;
import camml.core.models.ModelLearner;
import camml.core.models.cpt.CPTLearner;
import camml.core.models.dual.DualLearner;
import camml.core.models.logit.LogitLearner;
import camml.core.search.SearchPackage;

/**Interface to define search parameters for the GUI.
 * These could all be defined in GUIModel or elsewhere, but are done here for clarity
 * and ease of maintenance.
 * 
 */
public interface GUIParameters {
	
	//TODO: Version number 
	public String versionNumber = "1.00";
	
	//Formatting options for results table
	public static final DecimalFormat formatPosterior = new DecimalFormat("0.00000");		//Posteriors to 5 DP
	public static final DecimalFormat formatRelativePrior = new DecimalFormat("0.0000");	//Relative priors to 4DP
	public static final DecimalFormat formatBestMML = new DecimalFormat("0.000");			//3 DP
	public static final DecimalFormat formatWeight = new DecimalFormat("0.000");			//3 DP
	
	//Minumum and maximum allowable values for search factor, maxSECs and minTotalPosterior.
	public static final double minSearchFactor = 0.05;			//Minimum search factor value allowable
	public static final double maxSearchFactor = 50.0;			//Maximum search factor value allowable
	
	public static final int minSECs = 3;						//Minimum value for "maxSECs" that is considered valid
	public static final int maxSECs = 1000;						//Maximum value for "maxSECs" that is considered valid
	
	public static final double min_minTotalPosterior = 0.30;		//Minimum value for "minTotalPosterior" that is considered valid
	public static final double max_minTotalPosterior = 1.00;		//Maximum value for "minTotalPosterior" that is considered valid
	
    public static final double minAlpha = 0.01;   //Minimum alpha value for conditional dependency test in latent variable detection
    public static final double maxAlpha = 0.99;   //Maximum alpha value for conditional dependency test in latent variable detection
	
    public static final double minErrorRate = 0.0;   //Minimum error rate for detecting latent variable
    public static final double maxErrorRate = 1;   //Maximum error rate for detecting latent variable
    
    
    public static final int minEMIteration = 10;     //Minimum number of iteration for EM in latent variable learning
    public static final int maxEMIteration = 10000;  //Maximum number of iteration for EM in latent variable learning
    
    public static final double minEMThreshold = 0.00000000001;  //Minimum value of terminating EM in latent variable learning
    public static final double maxEMThreshold = 0.1;       //Maximum value of terminating EM in latent variable learning
    
    public static final double minLatentArity = 2;   //Minimum value of latent variable arity (number of states)
    public static final double maxLatentArity = 6;  //Maximum value of latent variable arity (number of states)
    
    
	//Available search types: These options will be presented to user in combobox.
	//Note: MMLLearners.length must equal MLLLearnerNames.length
	public static final ModelLearner[] MMLLearners = {
			SearchPackage.mmlCPTLearner,
			SearchPackage.mmlLatentCPTLearner,
			SearchPackage.dTreeLearner,
			LogitLearner.logitLearner,
			DualLearner.dualCPTDTreeLearner,
			DualLearner.dualCPTLogitLearner,
			DualLearner.dualDTreeLogitLearner,
			DualLearner.dualCPTDTreeLogitLearner,
			CPTLearner.mlMultinomialCPTLearner
	};
	
	//Names associated with the above learners.
	public static final String[] MMLLearnerNames = {
			"MML: CPT",
			"MML: CPT Latent",
			"MML: DTree",
			"MML: Logit",
			"MML: CPT + DTree",
			"MML: CPT + Logit",
			"MML: DTree + Logit",
			"MML: CPT + DTree + Logit",
			"Max. Likelihood: CPT",
	};
	
	//ML Learner used for all instances of MetropolisSearch
	public static final ModelLearner MLLearner = CPTLearner.mlMultinomialCPTLearner;
	
	public static final String[] LatentInitialisationOptions = {
	        "latent as root",
	        "using dependencies",
	        "random"
	};
	
	
	/*Available Random Number Generators
	 * Presently: 2 each of Java/Wallace random, one for random seeds and
	 *  the other for set seeds.
	 * Note: RNGs.length must equal RNGsString.length
	 */
	public static final Random[] RNGs = {
		new Random(),
		new Random(),
		new WallaceRandom(),
		new WallaceRandom()
	};
	
	//To display in the GUI combo box:
	public static final String[] RNGsString = {
		"Java RNG - Random Seed",
		"Java RNG - Set Seed",
		"Wallace RNG - Random Seed",
		"Wallace RNG - Set Seeds"
	};
	
	//For the above RNG, should a set seed be used (or a random one instead?)
	//Used by GUI to enable/disable RNG seed textboxes.
	//Note: RNGUseSetSeed.length must equal RNGs.length
	public static final boolean[] RNGUseSetSeed = {
		false,
		true,
		false,
		true
	};
	
	//Wallace RNGs require 2 seeds. True if RNG requires 2nd seed, false otherwise.
	//Note: RNGUseSetSeed2.length must equal RNGs.length
	public static final boolean[] RNGUseSetSeed2 = {
		false,
		false,
		false,
		true
	};
	
	
	
	/*Default string for expert priors, when user presses "New" button
	 * (Also when the user first checks the 'use expert priors' checkbox)
	 */
	public static final String defaultNewExpertPriorString = 
			"set {" +
			"\n\t//Can specify number of variables in data set: i.e. 'n=10;'" +
			"\n\t//May specify tier prior (defaults to 1.0): i.e. 'tierPrior = 0.9;'" +
			"\n\t//May specify edit distance (ed) prior (defaults to ~0.73)" +
			"\n\t//May specify KT prior (defaults to ~0.73)" +
			"\n}" +
			"\ned {" +
			"\n\t//Prior in format of edit distance from specified network or part" +
			"\n\t// network. Prior based on edit distance from this network." +
			"\n\t//Note: This section optional. Can be removed if not used." +
			"\n\t// Example: To specify a diamond network we could use" +
			"\n\t// 'a -> b; a -> c; b -> d; c -> d;' or 'a -> b c; d <- b c;'" +
			"\n}" +
			"\n\n//Use kt { ... ) for Kendall Tau (KT) prior." +
			"\n//KT prior - effectively 'bubble sort distance' between two total orderings." +
			"\n//Minimal KT distance added to undirected edit distance to determine prior for" +
			"\n//a given structure." +
			"\n\ntier {" +
			"\n\t//Tiers allow a total ordering of variables to be specified." +
			"\n\t//Format: 'A B C < D E F < G H I;' means variables A,B,C are before" +
			"\n\t// D,E,F in the total ordering of variables, (i.e. A can be a parent" +
			"\n\t// of D, but D cannot be a parent of A) and so on." +
			"\n}" +
			"\narcs {" +
			"\n\t//Allows individual arc relationships to be specified." +
			"\n\t//Available arc types. Note: specified in format 'A -> B 0.7;'" +
			"\n\t// where the number is the probability of that arc existing." +
			"\n\t//Directed arc: i.e. A -> B or B <- A" +
			"\n\t//Undirected arc: i.e. A -- B" +
			"\n\t//Ancestor: A => B or B <= A" +
			"\n\t//Correlated: A == B" +
			"\n\t//Tier: A << B or B >> A" +
			"\n}";
	
}
