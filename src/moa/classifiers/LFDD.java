
/*
 *    LFDD.java
 * 
 *    @author Jorge Chamby-Diaz (jchambyd at gmail dot com)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import moa.classifiers.meta.HEFT;
import moa.core.Measurement;
import moa.options.ClassOption;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.FCBFSearch;
import weka.attributeSelection.SymmetricalUncertAttributeSetEval;

/**
 * LFDD: Landmark-based feature drift detector
 *
 * <p>See details in:<br> Jean Paul Barddal, Heitor Murilo Gomes, Fabrício 
 * Enembreck, Bernhard Pfahringer. A survey on feature drift adaptation: 
 * Definition, benchmark, challenges and future directions. In System and 
 * Software, DOI: 10.1016/j.jss.2016.07.005, ELSEVIER, 2017.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Base Classiﬁer to train.</li>
 * <li>-c : Chunk size</li>
 * </ul>
 *
 * @author Jorge Chamby-Diaz (jchambyd at gmail dot com)
 * @version $Revision: 1 $
 */
public class LFDD extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
        "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption chunkSizeOption = new IntOption("chunkSize", 'c',
        "The chunk size used for classifier creation and evaluation.", 500, 1, Integer.MAX_VALUE);
    
    protected Classifier classifier;
    protected AttributeSelection attSelector;
    protected Instances buffer;
    protected weka.core.Instances wbuffer;
    protected SamoaToWekaInstanceConverter convToWeka;
    protected WekaToSamoaInstanceConverter convToMoa;
    protected int[] currentAttr;

    @Override
    public void resetLearningImpl()
    {
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();
        this.buffer = null;
        this.convToWeka = new SamoaToWekaInstanceConverter();
        this.currentAttr = new int[0];
    }

    @Override
    public void trainOnInstanceImpl(Instance inst)
    {
        // Store instance in the buffer
        if (this.buffer == null) {
            this.buffer = new Instances(inst.dataset());
        }

        double mt = this.buffer.numInstances();
        Instance trainInst = inst.copy();

        // Chunk is not full
        if (mt != this.chunkSizeOption.getValue()) {
            this.buffer.add(inst);
        } else {
            // Feature Selection performed over data chunck
            this.attSelector = this.performFeatureSelection();
            int[] newAttr = null;
            try {
                newAttr = this.attSelector.selectedAttributes();
            } catch (Exception ex) {
                Logger.getLogger(LFDD.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            Arrays.sort(newAttr);
            if(!Arrays.equals(this.currentAttr, newAttr))
            {
                this.classifier.resetLearning();
                this.convToMoa = new WekaToSamoaInstanceConverter();
                this.currentAttr = newAttr;
            }
            this.buffer = new Instances(this.getModelContext());
        }
        if(this.attSelector != null)
        {
            weka.core.Instance winst = this.convToWeka.wekaInstance(inst);
            try {
                trainInst = this.convToMoa.samoaInstance(this.attSelector.reduceDimensionality(winst));
            } catch (Exception ex) {
                Logger.getLogger(LFDD.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        this.classifier.trainOnInstance(trainInst);        
    }

    @Override
    public boolean isRandomizable()
    {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance inst)
    {
        Instance trainInst = inst.copy();

        try {
            if(this.attSelector != null){
                weka.core.Instance winst = this.convToWeka.wekaInstance(inst);
                trainInst = this.convToMoa.samoaInstance(this.attSelector.reduceDimensionality(winst));
            }            
        } catch (Exception ex) {
            Logger.getLogger(LFDD.class.getName()).log(Level.SEVERE, null, ex);
        }
        return this.classifier.getVotesForInstance(trainInst);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent)
    {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl()
    {
        Measurement[] measurements = null;
        return measurements;
    }

    private AttributeSelection performFeatureSelection()
    {
        this.wbuffer = this.convToWeka.wekaInstances(this.buffer);
        AttributeSelection attsel = new AttributeSelection();
        SymmetricalUncertAttributeSetEval evaluator = new SymmetricalUncertAttributeSetEval();

        attsel.setEvaluator(evaluator);
        FCBFSearch search = new FCBFSearch();
        search.setThreshold(0);
        attsel.setSearch(search);

        try {
            attsel.SelectAttributes(this.wbuffer);
        } catch (Exception ex) {
            Logger.getLogger(HEFT.class.getName()).log(Level.SEVERE, null, ex);
        }

        return attsel;
    }
}