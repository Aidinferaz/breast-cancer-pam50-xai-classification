"""
Biological Validation Module

This module provides tools for validating machine learning results against
biological databases and literature, specifically for breast cancer PAM50 classification.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional
import warnings


class BiologicalValidator:
    """
    Validate ML-identified genes against biological databases and literature.
    """
    
    def __init__(self):
        """Initialize validator with reference gene sets."""
        self.pam50_genes = self._load_pam50_genes()
        self.subtype_markers = self._load_subtype_markers()
        self.oncogenes = self._load_oncogenes()
        self.tumor_suppressors = self._load_tumor_suppressors()
    
    def _load_pam50_genes(self) -> Set[str]:
        """
        Load original PAM50 gene list from Parker et al. 2009.
        
        Reference:
        Parker, J. S., et al. (2009). "Supervised risk predictor of breast cancer 
        based on intrinsic subtypes." Journal of Clinical Oncology, 27(8), 1160-1167.
        """
        pam50_genes = [
            'ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 
            'CCNE1', 'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5',
            'EGFR', 'ERBB2', 'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1',
            'GPR160', 'GRB7', 'KIF2C', 'KRT14', 'KRT17', 'KRT5', 'MAPT',
            'MDM2', 'MELK', 'MIA', 'MKI67', 'MLPH', 'MMP11', 'MYBL2',
            'MYC', 'NAT1', 'ORC6', 'PGR', 'PHGDH', 'PTTG1', 'RRM2',
            'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS', 'UBE2C', 'UBE2T'
        ]
        return set(pam50_genes)
    
    def _load_subtype_markers(self) -> Dict[str, Set[str]]:
        """
        Load known markers for each PAM50 subtype.
        
        Based on published literature and cancer genomics databases.
        """
        markers = {
            'Luminal A': {
                'ESR1', 'PGR', 'FOXA1', 'XBP1', 'GATA3', 'BCL2', 'NAT1', 
                'SLC40A1', 'MLPH', 'BAG1', 'MAPT'
            },
            'Luminal B': {
                'ESR1', 'MKI67', 'CCNB1', 'MYBL2', 'FGFR4', 'CCNE1', 
                'GRB7', 'ERBB2', 'UBE2C', 'RRM2'
            },
            'HER2-enriched': {
                'ERBB2', 'GRB7', 'STARD3', 'PGAP3', 'PNMT', 'TCAP',
                'TMEM45B', 'CDC6', 'UBE2T', 'BIRC5'
            },
            'Basal-like': {
                'FOXC1', 'KRT5', 'KRT17', 'KRT14', 'EGFR', 'CDH3', 
                'MMP11', 'ANLN', 'CENPF', 'MELK', 'PTTG1'
            },
            'Normal-like': {
                'SFRP1', 'CXXC5', 'ACTR3B', 'GPR160', 'MIA'
            }
        }
        return markers
    
    def _load_oncogenes(self) -> Set[str]:
        """
        Load known oncogenes relevant to breast cancer.
        
        Source: COSMIC Cancer Gene Census and literature.
        """
        oncogenes = {
            'ERBB2', 'MYC', 'CCND1', 'PIK3CA', 'AKT1', 'EGFR', 'FGFR1',
            'FGFR2', 'MDM2', 'CCNE1', 'KRAS', 'CDK4', 'CDK6', 'MET'
        }
        return oncogenes
    
    def _load_tumor_suppressors(self) -> Set[str]:
        """
        Load known tumor suppressor genes relevant to breast cancer.
        
        Source: COSMIC Cancer Gene Census and literature.
        """
        tumor_suppressors = {
            'TP53', 'BRCA1', 'BRCA2', 'PTEN', 'RB1', 'CDH1', 'ATM',
            'CHEK2', 'PALB2', 'STK11', 'NF1', 'MAP3K1'
        }
        return tumor_suppressors
    
    def check_pam50_overlap(self, genes: List[str]) -> Dict:
        """
        Check overlap with original PAM50 genes.
        
        Parameters:
        -----------
        genes : list
            List of gene symbols to check
        
        Returns:
        --------
        overlap_info : dict
            Dictionary with overlap statistics
        """
        genes_set = set(genes)
        overlap = genes_set.intersection(self.pam50_genes)
        novel_genes = genes_set - self.pam50_genes
        
        overlap_info = {
            'total_input_genes': len(genes),
            'overlap_count': len(overlap),
            'overlap_genes': sorted(list(overlap)),
            'overlap_percentage': (len(overlap) / len(genes) * 100) if len(genes) > 0 else 0,
            'novel_genes': sorted(list(novel_genes)),
            'novel_count': len(novel_genes)
        }
        
        return overlap_info
    
    def validate_subtype_markers(self, genes: List[str], subtype: str) -> Dict:
        """
        Validate if genes are known markers for a specific subtype.
        
        Parameters:
        -----------
        genes : list
            List of gene symbols
        subtype : str
            PAM50 subtype name
        
        Returns:
        --------
        validation : dict
            Validation results
        """
        # Normalize subtype name
        subtype_map = {
            'luma': 'Luminal A',
            'luminal a': 'Luminal A',
            'luminal_a': 'Luminal A',
            'lumb': 'Luminal B',
            'luminal b': 'Luminal B',
            'luminal_b': 'Luminal B',
            'her2': 'HER2-enriched',
            'her2-enriched': 'HER2-enriched',
            'basal': 'Basal-like',
            'basal-like': 'Basal-like',
            'normal': 'Normal-like',
            'normal-like': 'Normal-like'
        }
        
        subtype = subtype_map.get(subtype.lower(), subtype)
        
        if subtype not in self.subtype_markers:
            warnings.warn(f"Unknown subtype: {subtype}")
            return {'error': f'Unknown subtype: {subtype}'}
        
        genes_set = set(genes)
        known_markers = self.subtype_markers[subtype]
        
        validated_genes = genes_set.intersection(known_markers)
        unvalidated_genes = genes_set - known_markers
        
        validation = {
            'subtype': subtype,
            'total_genes': len(genes),
            'validated_count': len(validated_genes),
            'validated_genes': sorted(list(validated_genes)),
            'validated_percentage': (len(validated_genes) / len(genes) * 100) if len(genes) > 0 else 0,
            'unvalidated_genes': sorted(list(unvalidated_genes)),
            'known_markers_for_subtype': sorted(list(known_markers))
        }
        
        return validation
    
    def check_cancer_relevance(self, genes: List[str]) -> Dict:
        """
        Check if genes are known oncogenes or tumor suppressors.
        
        Parameters:
        -----------
        genes : list
            List of gene symbols
        
        Returns:
        --------
        relevance : dict
            Cancer relevance information
        """
        genes_set = set(genes)
        
        oncogene_hits = genes_set.intersection(self.oncogenes)
        tumor_suppressor_hits = genes_set.intersection(self.tumor_suppressors)
        
        relevance = {
            'oncogenes': sorted(list(oncogene_hits)),
            'oncogene_count': len(oncogene_hits),
            'tumor_suppressors': sorted(list(tumor_suppressor_hits)),
            'tumor_suppressor_count': len(tumor_suppressor_hits),
            'known_cancer_genes': sorted(list(oncogene_hits | tumor_suppressor_hits)),
            'known_cancer_gene_count': len(oncogene_hits | tumor_suppressor_hits),
            'novel_genes': sorted(list(genes_set - oncogene_hits - tumor_suppressor_hits))
        }
        
        return relevance
    
    def validate_gene_list(self, genes: List[str], verbose: bool = True) -> pd.DataFrame:
        """
        Comprehensive validation of a gene list.
        
        Parameters:
        -----------
        genes : list
            List of gene symbols
        verbose : bool, default=True
            Print summary
        
        Returns:
        --------
        validation_df : DataFrame
            Validation results for each gene
        """
        results = []
        
        for gene in genes:
            # Check PAM50 membership
            in_pam50 = gene in self.pam50_genes
            
            # Check if oncogene or tumor suppressor
            is_oncogene = gene in self.oncogenes
            is_tumor_suppressor = gene in self.tumor_suppressors
            
            # Check which subtypes it's a marker for
            marker_for_subtypes = []
            for subtype, markers in self.subtype_markers.items():
                if gene in markers:
                    marker_for_subtypes.append(subtype)
            
            # Determine biological evidence level
            if in_pam50:
                evidence = 'Strong (PAM50)'
            elif is_oncogene or is_tumor_suppressor:
                evidence = 'Strong (Cancer Gene)'
            elif marker_for_subtypes:
                evidence = 'Moderate (Subtype Marker)'
            else:
                evidence = 'Unknown'
            
            results.append({
                'gene': gene,
                'in_pam50': in_pam50,
                'oncogene': is_oncogene,
                'tumor_suppressor': is_tumor_suppressor,
                'subtype_marker_for': ', '.join(marker_for_subtypes) if marker_for_subtypes else 'None',
                'biological_evidence': evidence
            })
        
        validation_df = pd.DataFrame(results)
        
        if verbose:
            print("\n" + "=" * 70)
            print("BIOLOGICAL VALIDATION SUMMARY")
            print("=" * 70)
            print(f"Total genes: {len(genes)}")
            print(f"PAM50 genes: {validation_df['in_pam50'].sum()}")
            print(f"Oncogenes: {validation_df['oncogene'].sum()}")
            print(f"Tumor suppressors: {validation_df['tumor_suppressor'].sum()}")
            print(f"\nEvidence levels:")
            print(validation_df['biological_evidence'].value_counts())
            print("=" * 70)
        
        return validation_df
    
    def generate_validation_report(self, 
                                   top_genes_by_subtype: Dict[str, List[str]],
                                   save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report for multiple subtypes.
        
        Parameters:
        -----------
        top_genes_by_subtype : dict
            Dictionary of {subtype: [genes]}
        save_path : str, optional
            Path to save report
        
        Returns:
        --------
        report : str
            Formatted validation report
        """
        lines = []
        lines.append("# Biological Validation Report")
        lines.append("# Breast Cancer PAM50 Subtype Classification\n")
        lines.append("=" * 80)
        lines.append("\n## Overview\n")
        
        total_genes = sum(len(genes) for genes in top_genes_by_subtype.values())
        lines.append(f"- Total subtypes analyzed: {len(top_genes_by_subtype)}")
        lines.append(f"- Total genes identified: {total_genes}\n")
        
        # Per-subtype analysis
        for subtype, genes in top_genes_by_subtype.items():
            lines.append("\n" + "=" * 80)
            lines.append(f"\n## {subtype} Subtype\n")
            lines.append("-" * 80)
            
            # PAM50 overlap
            overlap = self.check_pam50_overlap(genes)
            lines.append(f"\n### PAM50 Gene Overlap\n")
            lines.append(f"- Total genes: {overlap['total_input_genes']}")
            lines.append(f"- PAM50 overlap: {overlap['overlap_count']} genes "
                        f"({overlap['overlap_percentage']:.1f}%)")
            
            if overlap['overlap_genes']:
                lines.append(f"- PAM50 genes found: {', '.join(overlap['overlap_genes'])}")
            
            if overlap['novel_genes']:
                lines.append(f"- Novel genes: {overlap['novel_count']}")
            
            # Subtype marker validation
            subtype_validation = self.validate_subtype_markers(genes, subtype)
            if 'error' not in subtype_validation:
                lines.append(f"\n### Known Subtype Markers\n")
                lines.append(f"- Validated markers: {subtype_validation['validated_count']} "
                           f"({subtype_validation['validated_percentage']:.1f}%)")
                
                if subtype_validation['validated_genes']:
                    lines.append(f"- Validated: {', '.join(subtype_validation['validated_genes'])}")
            
            # Cancer gene relevance
            cancer_relevance = self.check_cancer_relevance(genes)
            lines.append(f"\n### Cancer Gene Relevance\n")
            lines.append(f"- Known cancer genes: {cancer_relevance['known_cancer_gene_count']}")
            lines.append(f"- Oncogenes: {cancer_relevance['oncogene_count']}")
            
            if cancer_relevance['oncogenes']:
                lines.append(f"  - {', '.join(cancer_relevance['oncogenes'])}")
            
            lines.append(f"- Tumor suppressors: {cancer_relevance['tumor_suppressor_count']}")
            
            if cancer_relevance['tumor_suppressors']:
                lines.append(f"  - {', '.join(cancer_relevance['tumor_suppressors'])}")
            
            # Detailed validation table
            validation_df = self.validate_gene_list(genes, verbose=False)
            lines.append(f"\n### Detailed Gene Validation\n")
            lines.append("```")
            lines.append(validation_df.to_string(index=False))
            lines.append("```\n")
        
        # Summary
        lines.append("\n" + "=" * 80)
        lines.append("\n## Overall Summary\n")
        
        all_genes = []
        for genes in top_genes_by_subtype.values():
            all_genes.extend(genes)
        
        all_overlap = self.check_pam50_overlap(all_genes)
        all_relevance = self.check_cancer_relevance(all_genes)
        
        lines.append(f"- Total unique genes across all subtypes: {len(set(all_genes))}")
        lines.append(f"- PAM50 overlap: {all_overlap['overlap_count']} "
                    f"({all_overlap['overlap_percentage']:.1f}%)")
        lines.append(f"- Known cancer genes: {all_relevance['known_cancer_gene_count']}")
        lines.append(f"- Novel genes requiring further investigation: "
                    f"{len(all_relevance['novel_genes'])}\n")
        
        lines.append("\n## Recommendations\n")
        lines.append("1. **High-confidence genes** (PAM50 or known cancer genes): "
                    "Use directly for clinical interpretation")
        lines.append("2. **Moderate-confidence genes** (subtype markers): "
                    "Validate with pathway analysis")
        lines.append("3. **Novel genes**: Perform literature search and pathway enrichment")
        lines.append("4. **External validation**: Test identified genes on independent datasets")
        lines.append("5. **Functional validation**: Consider experimental validation for novel genes\n")
        
        lines.append("=" * 80)
        
        report = '\n'.join(lines)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {save_path}")
        
        return report


def get_gene_description(gene: str) -> Dict[str, str]:
    """
    Get basic description for a gene (placeholder for future API integration).
    
    Parameters:
    -----------
    gene : str
        Gene symbol
    
    Returns:
    --------
    description : dict
        Gene description (placeholder)
    """
    # This is a placeholder. In practice, you would integrate with:
    # - NCBI Gene API
    # - Ensembl REST API
    # - MyGene.info API
    
    description = {
        'gene': gene,
        'status': 'Not implemented',
        'note': 'Integrate with NCBI Gene API or MyGene.info for detailed descriptions'
    }
    
    return description


def suggest_pathway_analysis_tools() -> None:
    """Print suggestions for pathway analysis tools."""
    print("\n" + "=" * 70)
    print("PATHWAY ENRICHMENT ANALYSIS RECOMMENDATIONS")
    print("=" * 70)
    print("\nFor comprehensive biological validation, consider using:")
    print("\n1. **Enrichr** (https://maayanlab.cloud/Enrichr/)")
    print("   - Web-based, easy to use")
    print("   - Multiple pathway databases (KEGG, GO, Reactome, WikiPathways)")
    print("   - Python API available: gseapy.enrichr()")
    
    print("\n2. **GSEA** (Gene Set Enrichment Analysis)")
    print("   - Gold standard for pathway analysis")
    print("   - Uses full ranked gene list")
    print("   - Python implementation: gseapy.prerank()")
    
    print("\n3. **DAVID** (Database for Annotation, Visualization and Integrated Discovery)")
    print("   - Comprehensive functional annotation")
    print("   - https://david.ncifcrf.gov/")
    
    print("\n4. **STRING** (https://string-db.org/)")
    print("   - Protein-protein interaction networks")
    print("   - Functional enrichment")
    
    print("\n5. **cBioPortal** (https://www.cbioportal.org/)")
    print("   - Cancer genomics data portal")
    print("   - Check expression in TCGA cohorts")
    print("   - Survival analysis")
    
    print("\nExample code:")
    print("```python")
    print("import gseapy as gp")
    print("# Enrichr analysis")
    print("enr = gp.enrichr(")
    print("    gene_list=['ERBB2', 'GRB7', 'STARD3'],")
    print("    gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021'],")
    print("    organism='Human'")
    print(")")
    print("print(enr.results.head())")
    print("```")
    print("=" * 70 + "\n")


def create_validation_checklist() -> str:
    """Generate a validation checklist for biological trustworthiness."""
    checklist = """
    # Biological Validation Checklist
    
    Use this checklist to ensure comprehensive validation of your ML results:
    
    ## 1. Gene-Level Validation
    - [ ] Check PAM50 overlap for all identified genes
    - [ ] Verify genes are known cancer-related (oncogenes/tumor suppressors)
    - [ ] Confirm genes are appropriate markers for their predicted subtype
    - [ ] Search PubMed for each gene + "breast cancer"
    - [ ] Check expression patterns in cBioPortal TCGA-BRCA cohort
    
    ## 2. Pathway-Level Validation
    - [ ] Perform pathway enrichment analysis (Enrichr, GSEA)
    - [ ] Verify enriched pathways are biologically relevant
    - [ ] Check for cancer hallmark pathways (proliferation, apoptosis, etc.)
    - [ ] Examine protein-protein interactions (STRING)
    
    ## 3. Literature Validation
    - [ ] Systematic PubMed search for top genes
    - [ ] Review recent breast cancer subtyping papers
    - [ ] Check genes against COSMIC Cancer Gene Census
    - [ ] Verify with breast cancer review articles
    
    ## 4. Data Validation
    - [ ] Check gene expression patterns make biological sense
    - [ ] Verify batch effects are not driving results
    - [ ] Confirm features are not technical artifacts
    - [ ] Validate on independent external dataset
    
    ## 5. Clinical Relevance
    - [ ] Consult with clinicians/pathologists
    - [ ] Check if genes have clinical actionability
    - [ ] Verify alignment with current clinical guidelines
    - [ ] Consider prognostic/predictive value
    
    ## 6. Documentation
    - [ ] Document all validation sources
    - [ ] Create comprehensive validation report
    - [ ] List assumptions and limitations
    - [ ] Include references for all claims
    
    ## Success Criteria
    - At least 50% of top genes should have PAM50 or literature support
    - Pathway analysis should show biologically coherent themes
    - External validation accuracy should be within 10% of training
    - Clinical experts should find results interpretable and useful
    """
    return checklist
