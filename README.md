<!-- Output copied to clipboard! -->

<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 0.587 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β29
* Tue Feb 23 2021 21:20:43 GMT-0800 (PST)
* Source doc: HR-attrition
* Tables are currently converted to HTML tables.
----->


[https://sdv.dev/SDV/user_guides/evaluation/evaluation_framework.html](https://sdv.dev/SDV/user_guides/evaluation/evaluation_framework.html) 



*   "The output of this function call will be a number between 0 and 1 that will indicate us how similar the two tables are, being 0 the worst and 1 the best possible score."
    *   This is incorrect even in the given documentation example
*   bugs
    *   sdv parameter names for copulaGAN had to be updated
    *   the ord_feats had to be fixed
    *   "\r" in the raw ipynb file causes an editor crash in jupyter notebook, I removed all of them in a python script
*   Methodology Issues
    *   He used AUC to choose his first model which was lr
    *   Then he used AUC to choose his last model which was catboost, but he chose gbc which had the second highest AUC
        *   I tried gbc with synthetic + original data and with only original data and found you get higher results with synthetic + original data
    *   Dataset differences
        *   the file size is smaller for the dataset given compared to the kaggle ibm one that is linked.
        *   Both had a dimension of (1470, 35) so I think the difference is the compression algorithm from storing the data on github

<table>
  <tr>
   <td>
data
   </td>
   <td>Classifier
   </td>
   <td>Accuracy
   </td>
   <td>AUC
   </td>
   <td>Recall
   </td>
   <td>Precision
   </td>
   <td>F1
   </td>
   <td>Kappa
   </td>
   <td>MCC
   </td>
  </tr>
  <tr>
   <td>Original
   </td>
   <td>lr
   </td>
   <td>0.8794	
   </td>
   <td>0.8534	
   </td>
   <td>0.4463	
   </td>
   <td>0.7006	
   </td>
   <td>0.5388	
   </td>
   <td>0.4746	
   </td>
   <td>0.4934
   </td>
  </tr>
  <tr>
   <td>Original + synth
   </td>
   <td>lr
   </td>
   <td>0.8971	
   </td>
   <td>0.9564	
   </td>
   <td>0.8420	
   </td>
   <td>0.9512	
   </td>
   <td>0.8562	
   </td>
   <td>0.7964	
   </td>
   <td>0.8200
   </td>
  </tr>
  <tr>
   <td>Original
   </td>
   <td>gbc
   </td>
   <td>0.8686	
   </td>
   <td>0.8195	
   </td>
   <td>0.3140
   </td>
   <td>0.7010
   </td>
   <td>0.4233	
   </td>
   <td>0.3648	
   </td>
   <td>0.4056
   </td>
  </tr>
  <tr>
   <td>Original + synth
   </td>
   <td>gbc
   </td>
   <td>0.8971
   </td>
   <td>0.9564
   </td>
   <td>0.8420
   </td>
   <td>0.9512
   </td>
   <td>0.8562
   </td>
   <td>0.7964
   </td>
   <td>0.8200
   </td>
  </tr>
</table>


lr = logistic regression

gbc = Gradient boosting classifier
