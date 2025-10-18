global_output_max_new_tokens = 1024 #512 #2048

pick_unique_sentences = """"""

llm_judge_prompt2 = """You are an expert in natural language processing. Given an extracted entity-relation triple from it, your task is to:
                                1. Evaluate the **Relevance** of the triple to understanding the disease ***GNE Myopathy***.
                                2. Assess the **Tag and Relationship Accuracy** to determine if the triple makes logical sense and if the entity tags are appropriate.
                                3. Assign a score from 1 to 5 based on these criteria

                                Use a scoring scale from 1 to 5, where:
                                - **5**: Highly relevant and correct
                                - **4**: Mostly relevant and correct with minor issues
                                - **3**: Somewhat relevant or partially correct
                                - **2**: Poorly relevant or mostly incorrect
                                - **1**: Completely irrelevant or incorrect

                                The final output or response with only be the numerical score.
                    """

llm_judge_prompt = """-Goal-
                            Given a context sentence and a tagged triplet, assess the relevance of each entity, relationship, and tag to GNE myopathy. Rate the overall relevance on a scale from 1 to 5, where 5 indicates high relevance and 1 indicates low relevance.

                            Relevance should be determined by the importance of the entity or relationship in understanding GNE myopathy within medical and biological contexts. Consider factors such as disease symptoms, affected body parts, biological markers, diagnostic and treatment methods, and research on the disease.

                            -Tagging Categories-
                            Entities in the triplet should fall under one of the following categories:
                            "location", "institute", "cell component", "disease", "body part", "enzyme", "biological process", "sample", "population group", "treatment", "symptom", "protein", "gene", "pathway", "mutation", "diagnostic procedure", "drug", "feature", "model", "chemical", "time period", "observation", "role".

                            -Steps-
                            1. Review the context sentence and the triplet to understand the relationship and entities in the context of GNE myopathy.
                            2. Verify that the two entities of the triplet are correctly tagged based on the provided categories, ensuring that each term accurately reflects its role within the context of sentence and GNE myopathy.
                            3. Assign a relevance score from 1 to 5 based on three criteria: (1) the relevance of the triplet to GNE myopathy, (2) the clarity and appropriateness of the meanings in the context sentence, and (3) the accuracy of the tagging for each term.

                            -Output Format-
                            Numerical score 

                            ################
                            -Examples-
                            ################

                            Example Input:
                            "Patients with GNE myopathy often experience scapular weakness, a common feature associated with the progression of this disease. -(GNE Myopathy (disease), HAS FEATURE, Scapular weakness (symptom))"

                            Expected Output:
                            5 

                            Example Input: 
                            "The introduction of new research highlights the significance of the GNE gene in muscle disorders. -(Introduction (feature), PRESENT IN, the GNE gene (gene))"

                            Expected Output: 
                            2 
"""



tagging_triples_prompt_with_relevant_sentences = """-Goal-
                                Given a context sentence and a triplet extracted from it, categorize the first and third terms based on the following categories and add the category in parentheses next to each term.

                                Possible categories include:
                                "location", "institute", "cell component", "disease", "body part", "enzyme", "biological process", "sample", "population group", "treatment", "symptom", "protein", "gene", "pathway", "mutation", "diagnostic procedure", "drug", "feature", "model", "chemical", "time period", "observation" and "role"
                                
                                Donot include entities that can fall in these categories:
                                "email", "author"
                                
                                -Steps-
                                1. Identify the first and third terms in the triplet present after - of the context sentence  
                                
                                2. Determine the appropriate category for each entity based on its meaning in the context sentence, using the provided category list.

                                2. Format the triplet by appending each terms category in parentheses immediately following the term.

                                -Output Format-
                                Display each triplet in the format:
                                (first term (category), relationship, third term (category))

                                ################
                                -Examples-
                                ################

                                Example Input:
                                "Patients with GNE myopathy often experience scapular weakness, a common feature associated with the progression of this disease.  - GNE Myopathy, HAS FEATURE, Scapular weakness"

                                Expected Output:
                                (GNE Myopathy (disease), HAS FEATURE, Scapular weakness (symptom))


"""


tagging_triples_prompt = """-Goal-
                                Given a triplet, categorize the first and third terms based on the following categories and add the category in parentheses next to each term.

                                Possible categories include:
                                "location", "institute", "cell component",  "disease", "body part", "enzyme", "biological process", "sample", "population group", "treatment", "symptom", "protein", "gene", "pathway", "mutation", "diagnostic procedure", "drug", "feature", "model", "chemical", "time period", "observation" and "role"
                                
                                Donot include entities that can fall in these categories:
                                "email", "author"
                                
                                -Steps-
                                1. Identify the first and third terms in the triplet and assign the appropriate category from the provided list.

                                2. Format the triplet by appending each terms category in parentheses immediately following the term.

                                -Output Format-
                                Display each triplet in the format:
                                (first term (category), relationship, third term (category))

                                ################
                                -Examples-
                                ################

                                Example Input:
                                "GNE Myopathy, HAS FEATURE, Scapular weakness"

                                Expected Output:
                                (GNE Myopathy (disease), HAS FEATURE, Scapular weakness (symptom))


"""




triplets_from_simple_sentence_prompt = """-Goal-
                                            Given a complex sentence, extract all entity pair relation triples from it. Each extracted triples should consist of two related entities and describe their relationship. 
                                            Only use information from the given text. 
            
                                            -Steps-
                                            Extract Entities: From the given sentence, identify relevant entities. Ensure each extracted entity retains the original context and meaning.

                                            entity_1: The first related entity.
                                            entity_2: The second related entity.
                                            Formulate Relationships: For each pair of related entities, define a one or two word phrase that describes their relationship while retaining the original context and meaning.

                                            relationship_description: One-two word explanation of how entity_1 and entity_2 are related.
                                            Output Format: Return the extracted entity pair relations in a list format, ensuring each relation conveys the original meaning.

                                            Donot generate multiple entity1-relation-entity2 triples with the same meaning.
                                            DO NOT GENERATE TRIPLES IF THERE IS NO RELEVANT INFORMATION IN THE INPUT.
                                            ONLY GENERATE TRIPLES RELATED TO MEDICAL/ BIOLOGICAL CONTEXT.
                                            If the sentence explicitly negates a relation, indicate the negation by using a relation like DOES_NOT_<ACTION>.

                                            Use a new line as the list delimiter. Only output the triples as response in this format and nothing else.

                                            ################
                                            -Examples-
                                            ################

                                            Example Input:
                                            "Inborn errors of metabolism include d-2-hydroxyglutaric aciduria and l-2-hydroxyglutaric aciduria."
                                            
                                            Expected Output:
                                            (Inborn errors of metabolism, INCLUDE, d-2-hydroxyglutaric aciduria)
                                            (Inborn errors of metabolism, INCLUDE, l-2-hydroxyglutaric aciduria)

                                            Example Input:
                                            "Combined d,l-2-hydroxyglutaric aciduria is a biochemical variant of 2-hydroxyglutaric aciduria."

                                            Expected Output:
                                            (Combined d,l-2-hydroxyglutaric aciduria, IS A, biochemical variant)
                                            (2-hydroxyglutaric aciduria, HAS, biochemical variant)

                                            Example Input:
                                            "Gain-of-function mutations in IDH2 cause D-2-HG accumulation in some D-2-HGA patients."

                                            Expected Output:
                                            (Gain-of-function mutations, PRESENT IN, IDH2)
                                            (IDH2, CAUSE, D-2-HG accumulation)
                                            (D-2-HG accumulation, PRESENT IN, D-2-HGA patients)

                                            Example Input:
                                            "T1 values of calf muscles in GNE myopathy decreased with the gradual aggravation of fat infiltration."

                                            Expected Output:
                                            (Calf muscles, TESTED IN, GNE myopathy)
                                            (Calf muscles, HAVE, T1 values)
                                            (T1 values of calf muscles, MEASURED IN, GNE myopathy)
                                            (T1 values, DECREASED WITH, gradual aggravation of fat infiltration)
                                            
"""


simple_sentence_prompt = """-Goal-
                                Given a complex sentence, extract smaller, meaningful sentences that convey the same meaning as the original. Each extracted sentence should consist of two related entities and describe their relationship. 
                                The entities must be related to this list:
                                [drug, therapy, symptom, medication, treat, alleviate, relief, remedy, prevalence, 
                                disease, illness, sickness, disorder, syndrome, condition, ailment,
                                infectious, viral, bacterial, fungal, parasitic, chronic, acute, terminal,
                                progressive, degenerative, autoimmune, inflammatory, deficiency, genetic, hereditary,
                                malignant, benign, metastatic, outbreak, epidemic, pandemic,
                                sign, manifestation, pain, fever, fatigue, nausea, vomiting, diarrhea, rash,
                                swelling, inflammation, diagnosis, prognosis, screening,
                                cause, etiology, pathogen, virus, bacteria, fungus, parasite, congenital,
                                environmental, lifestyle, treatment, pharmaceutical, prescription, cure,
                                vaccine, vaccination, immunization, surgery, operation, procedure, intervention,
                                dose, dosage, administration, manage, control, ease, mitigate, prevent,
                                incidence, morbidity, mortality, fatality rate, risk factor, comorbidity, complication,
                                cardiovascular, respiratory, neurological, gastrointestinal, musculoskeletal, endocrine, renal, hepatic]
                                
                                Only use information from the given text. 
                                Avoid generating sentences with the same meaning.

                                -Steps-

                                Extract Entities: From the given complex sentence, identify all relevant entities. Ensure each extracted entity retains the original context and meaning.
                                entity_1: The first related entity.
                                entity_2: The second related entity.

                                Formulate Relationships: For each pair of related entities, create a meaningful sentence that describes their relationship which retains the original context and meaning.
                                relationship_description: A brief explanation of how entity_1 and entity_2 are related.

                                Output Format: Return the extracted sentences in the format given in example outputs, ensuring each sentence conveys the original meaning. Only output the response in this format and nothing else.

                                ################
                                -Examples-
                                ################
                                Example Input:
                                "Gregersen et al (1977) were the first to identify enantiomeric d- and l-2-hydroxyglutaric acids (D-2-HG and L-2-HG) as normal constituents of human urine.Three years later, two novel inborn errors of metabolism were simultaneously reported in the Journal of Inherited Metabolic Disease.Chalmers et al (1980) identified a patient with d-2-hydroxyglutaric aciduria (D-2-HGA), while Duran et al (1980) described a case of l-2-hydroxyglutaric aciduria (L-2-HGA), landmark publications that identified the metabolic hallmarks (D- and L-2-HG) in these disorders.Muntau et al (2000) described a third biochemical variant of 2-hydroxyglutaric aciduria (2-HGA) when they reported three patients with elevated urinary D- and L-2-HG, denoted by these authors as “combined d,l-2-hydroxyglutaric aciduria” (D,L-2-HGA)."

                                Expected Output:
                                ("D-2-HG and L-2-HG are enantiomeric forms of 2-hydroxyglutaric acids.")
                                ("D-2-HG and L-2-HG are normal constituents of human urine.")
                                ("Inborn errors of metabolism include d-2-hydroxyglutaric aciduria and l-2-hydroxyglutaric aciduria.")
                                ("D-2-HG is a metabolic hallmark in d-2-hydroxyglutaric aciduria.")
                                ("D-2-HGA is d-2-hydroxyglutaric aciduria.")
                                ("L-2-HG is a metabolic hallmark in l-2-hydroxyglutaric aciduria.")
                                ("L-2-HGA is l-2-hydroxyglutaric aciduria.")
                                ("Combined d,l-2-hydroxyglutaric aciduria is a biochemical variant of 2-hydroxyglutaric aciduria.")
                                ("D-2-HG and L-2-HG are elevated in combined d,l-2-hydroxyglutaric aciduria.")

                                Example Input:
                                "Major milestones in research on these disorders came with gene discovery: D2HGDH encoding d-2-hydroxyglutarate dehydrogenase (D-2-HGDH)(Achouri et al 2004) and L2HGDH encoding l-2-hydroxyglutarate dehydrogenase (L-2-HGDH)(Rzem et al 2004; Topcu et al 2004).In many D-2-HGA, and the majority of L-2-HGA patients, genetic characterization revealed pathogenic mutations in these genes (Struys et al 2005b; Steenweg et al 2010).Nonetheless, in fully one-half of D-2-HGA patients no mutations in D2HGDH were detected (Kranendijk et al 2010a).Subsequently, we described gain-of-function mutations in isocitrate dehydrogenase 2 (IDH2) which proved causative for the D-2-HG accumulation in previously unclassified D-2-HGA patients (Kranendijk et al 2010b).In sum, the preceding decade has provided tremendous advances in our understanding of the inborn 2-hydroxyglutaric acidurias, which will undoubtedly provide a solid foundation from which to develop novel and effective treatment strategies."

                                Expected Output:
                                ("D2HGDH encodes d-2-hydroxyglutarate dehydrogenase.")
                                ("L2HGDH encodes l-2-hydroxyglutarate dehydrogenase.")
                                ("Pathogenic mutations in D2HGDH and L2HGDH are found in many D-2-HGA and L-2-HGA patients.")
                                ("No mutations in D2HGDH were detected in half of D-2-HGA patients.")
                                ("Gain-of-function mutations in IDH2 cause D-2-HG accumulation in some D-2-HGA patients.")
                                ("Advances in understanding 2-hydroxyglutaric acidurias will help develop treatment strategies.") 

                                Example Input:
                                "Antiviral medications are used to treat viral infections, while bacterial diseases require antibiotic therapy and often present with symptoms like fever and inflammation that can be alleviated with proper medication."

                                Expected Output:
                                ("Antiviral medications treat viral infections.")
                                ("Bacterial diseases require antibiotic therapy.")
                                ("Bacterial diseases present with symptoms like fever.")
                                ("Bacterial diseases present with symptoms like inflammation.")
                                ("Proper medication alleviates symptoms like fever.")
                                ("Proper medication alleviates symptoms like inflammation.")

"""



