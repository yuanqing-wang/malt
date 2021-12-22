import os

def vina_docking(smiles, protein):
    """ Dock a molecule against a protein.

    Parameters
    ----------
    smiles : str
        Smiles of the molecule to be docked

    protein : str
        A PDB identifier or a filename.

    Returns
    -------
    float : negative docking score
    """
    assert isinstance(protein, str)

    # make temporary folder
    import tempfile
    with tempfile.TemporaryDirectory() as tempdirname:
        # prepare protein
        if ".pdb" not in protein:
            os.system(
                "cd %s && wget http://www.pdb.org/pdb/files/%s.pdb" % (
                    tempdirname,
                    protein,
                ) + " && mv %s.pdb protein.pdb" % protein,
            )

        else:
            os.system(
                "cp %s %s" % (protein, tempdirname),
            )

        os.system(
            "cd %s && prepare_receptor -r protein.pdb" % tempdirname,
        )

        # prepare ligand
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol)
        writer = Chem.PDBWriter("%s/ligand.pdb" % tempdirname)
        writer.write(mol)

        os.system(
            "cd %s && prepare_ligand -l ligand.pdb" % tempdirname
        )

        from vina import Vina
        v = Vina()
        v.set_receptor("%s/protein.pdbqt" % tempdirname)
        v.set_ligand_from_file("%s/ligand.pdbqt" % tempdirname)
        v.compute_vina_maps(center=[0, 0, 0], box_size=[100, 100, 100])
        v.dock()
        energy = -v.score()[0]
        return energy
