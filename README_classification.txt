Les Programmes s'exécutent les uns à la suite des autres :

_ classif_0 :   Paramétrages / Import des données / Features Engineering / Split Train/Test
				Visualisation des données
				
				-> Ce programme doit toujours être lancé entièrement 
	
	***** LES PARAMÉTRAGES DES CHEMINS D'ENTRÉE/SORTIE SONT A MODIFIER ICI (section A MODIFIER) *****
	
_ classif_2 : Entrainement des modèles, recherche de meilleurs paramètres

			  A la fin de chacun de ces programmes une section # MODELE SELECTIONNE permet
			  de sélectionner le meilleur modèle et d'enregistrer un PDF résumant les paramètres choisi 
			  et sa performance
			  
			  Les fichiers suffixé _gridsearch font une recherche de paramètres sur une grille
			  
_ classif_3 : Analyse Factorielle Multiple

			  Clustering des données d'entrainements & Analyse des 2 groupes trouvés
			  
	***** Ce fichier doit être lancé avant les programmes classif_4 (apprentissage expert)
			  qui entrainent 2 modèles sur chacun des groupes. *****
			  
_ classif_4 : Entrainement Experts

_ classif 5 : Export sur données test. Un programme pour les modèles sur jeu de données entier 
			  et un programmes pour export modèles experts sur les 2 groupes
			  
	***** LES PARAMÉTRAGES MODÈLE TESTÉ ET NOM DU FICHIER DE SORTIE SONT A MODIFIER ICI (section A MODIFIER) ******
	
	
Les lancement s'exécutent donc selon le plan :

classif_0 -> classif_2 -> classif_5 				pour les modèles classant les données à partir d'un groupe
classif_0 -> classif_3 -> classif_4 -> classif_5 	pour les modèles experts sur les 2 groupes		  


_export_model_report_pdf contient une fonction sortant un fichier PDF récapitulant les performances et les paramètres d'un modèle
_plot_cat_vs_quants et _spineplot_with_props contiennent des fonctions pour sortir des graphiques de la variable cible en fonction des variables catégorielles et quantitatives