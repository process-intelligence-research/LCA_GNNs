# Machine learning
# Standard library
import json
import os

import torch

# Machine learning
import torch.nn.functional as F

# Path management
import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT, HybridizationType

# Machine learning
from torch_geometric.data import Data, InMemoryDataset

# Machine learning
from torch_sparse import coalesce


class BaseMolecularDataset(InMemoryDataset):
    """
    Base class for molecular datasets with common functionality.

    This class provides shared utilities for molecular graph processing,
    atom/bond feature extraction, and common dataset operations.
    """

    # Shared atom types mapping
    types = {
        "Br": 0,
        "C": 1,
        "Ca": 2,
        "Cl": 3,
        "Cr": 4,
        "Cu": 5,
        "F": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "Li": 10,
        "Mn": 11,
        "N": 12,
        "Na": 13,
        "O": 14,
        "P": 15,
        "S": 16,
        "Si": 17,
        "Zn": 18,
    }

    # Shared bond types mapping
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, None: 4}

    # Class variables to cache mapping data
    _country_mapping = None
    _energy_mapping = None

    def __init__(
        self, root, dataexcel, transform=None, pre_transform=None, pre_filter=None
    ):
        self.dataexcel = dataexcel
        self.data_list = []  # Initialize data_list for GNN datasets
        super().__init__(root, transform, pre_transform, pre_filter)
        # Safe to use weights_only=False for our own processed data files
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @classmethod
    def _load_country_mapping(cls):
        """Load country mapping from JSON file."""
        if cls._country_mapping is None:
            # Get the path to the data directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "..", "..", "data", "raw")
            country_mapping_path = os.path.join(data_dir, "country_mapping.json")

            try:
                with open(country_mapping_path) as f:
                    cls._country_mapping = json.load(f)
            except FileNotFoundError:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Country mapping file not found at {country_mapping_path}"
                )
                cls._country_mapping = {}
        return cls._country_mapping

    @classmethod
    def _load_energy_mapping(cls):
        """Load energy mapping from JSON file."""
        if cls._energy_mapping is None:
            # Get the path to the data directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "..", "..", "data", "raw")
            energy_mapping_path = os.path.join(data_dir, "energy_mapping.json")

            try:
                with open(energy_mapping_path) as f:
                    cls._energy_mapping = json.load(f)
            except FileNotFoundError:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Energy mapping file not found at {energy_mapping_path}"
                )
                cls._energy_mapping = {}
        return cls._energy_mapping

    def download(self):
        """Placeholder download method as data is not publicly available."""
        pass

    def _extract_atom_features(self, mol):
        """
        Extract atom features from RDKit molecule.

        Returns atom feature tensors (x1, x2, x3) for concatenation.
        """
        type_idx = []
        aromatic = []
        ring = []
        sp = []
        sp2 = []
        sp3 = []
        sp3d = []
        sp3d2 = []
        num_neighbors = []

        for atom in mol.GetAtoms():
            type_idx.append(self.types[atom.GetSymbol()])
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            ring.append(1 if atom.IsInRing() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
            sp3d2.append(1 if hybridization == HybridizationType.SP3D2 else 0)
            num_neighbors.append(len(atom.GetNeighbors()))

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
        x2 = (
            torch.tensor([aromatic, ring, sp, sp2, sp3, sp3d, sp3d2], dtype=torch.float)
            .t()
            .contiguous()
        )
        x3 = F.one_hot(torch.tensor(num_neighbors), num_classes=6)

        return torch.cat([x1.to(torch.float), x2, x3.to(torch.float)], dim=-1)

    def _extract_bond_features(self, mol, N):
        """
        Extract bond features from RDKit molecule.

        Returns edge_index and edge_attr tensors.
        """
        row, col, bond_idx, conj, ring, stereo = [], [], [], [], [], []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_idx += 2 * [self.bonds[bond.GetBondType()]]
            conj.append(bond.GetIsConjugated())
            conj.append(bond.GetIsConjugated())
            ring.append(bond.IsInRing())
            ring.append(bond.IsInRing())
            stereo.append(bond.GetStereo())
            stereo.append(bond.GetStereo())

        edge_index = torch.tensor([row, col], dtype=torch.long)
        e1 = F.one_hot(torch.tensor(bond_idx), num_classes=len(self.bonds)).to(
            torch.float
        )
        e2 = torch.tensor([conj, ring], dtype=torch.float).t().contiguous()
        e3 = F.one_hot(torch.tensor(stereo), num_classes=6).to(torch.float)
        edge_attr = torch.cat([e1, e2, e3], dim=-1)

        return coalesce(edge_index, edge_attr, N, N)

    def _handle_single_atom_molecule(self):
        """
        Handle edge case of single-atom molecules.

        Returns default node features, edge_index, and edge_attr for single atoms.
        """
        print("Warning: molecule skipped because it contains only 1 atom")
        x = torch.tensor(
            [
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ],
            dtype=torch.float,
        )
        edge_index = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.long)
        edge_attr = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.long,
        )

        return x, edge_index, edge_attr

    def create_single_molecule_data(
        self,
        smiles: str,
        country_name: str = None,
        energy_mix: dict = None,
        dataset_type: str = "GNN_C",
    ) -> Data:
        """
        Create a single molecule data object from SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.
        country_name : str, optional
            Country name for GNN_C models or for energy mix lookup in GNN_E models.
        energy_mix : dict, optional
            Energy mix dictionary for GNN_E models. If not provided and country_name
            is given, will look up energy mix by country.
        dataset_type : str
            Type of dataset ("QSPR", "GNN_M", "GNN_C", "GNN_E").

        Returns
        -------
        Data
            PyTorch Geometric Data object for the molecule.
        """
        # Convert SMILES to molecular graph
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        mol = Chem.AddHs(mol)
        N = mol.GetNumAtoms()

        if N <= 1:
            x, edge_index, edge_attr = self._handle_single_atom_molecule()
        else:
            x = self._extract_atom_features(mol)
            edge_index, edge_attr = self._extract_bond_features(mol, N)

        # Create base data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Add dataset-specific features
        if dataset_type == "GNN_C":
            if country_name is None:
                raise ValueError("country_name must be provided for GNN_C models")
            # Convert country name to country_id
            country_id = self._get_country_id(country_name)
            country_onehot = F.one_hot(
                torch.tensor([country_id]), num_classes=91
            ).float()
            data.country_id = country_onehot

        elif dataset_type == "GNN_E":
            if energy_mix is not None:
                # Use provided energy mix dictionary
                energy_tensor = self._convert_energy_mix_to_tensor(energy_mix)
                data.energy_id = energy_tensor
            elif country_name is not None:
                # Look up energy mix by country name
                energy_tensor = self._get_energy_mix_by_country(country_name)
                data.energy_id = energy_tensor
            else:
                raise ValueError(
                    "Either energy_mix or country_name must be provided for GNN_E models"
                )

        return data

    def _get_country_id(self, country_name: str) -> int:
        """
        Convert country name to country ID using the mapping from JSON file.

        Parameters
        ----------
        country_name : str
            Name of the country.

        Returns
        -------
        int
            Country ID (0-90).
        """
        country_mapping = self._load_country_mapping()
        country_lower = country_name.lower()

        if country_lower in country_mapping:
            return country_mapping[country_lower]
        else:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Unknown country: {country_name}, using default ID 0")
            return 0

    def _convert_energy_mix_to_tensor(self, energy_mix: dict) -> torch.Tensor:
        """
        Convert energy mix dictionary to tensor.

        Parameters
        ----------
        energy_mix : dict
            Dictionary with energy mix values.
            Expected keys: ["Coal, peat and oil shale", "Crude, NGL and feedstocks",
                           "Oil products", "Natural gas", "Renewables and waste",
                           "Electricity", "Heat"]

        Returns
        -------
        torch.Tensor
            Tensor representation of energy mix.
        """
        expected_keys = [
            "Coal, peat and oil shale",
            "Crude, NGL and feedstocks",
            "Oil products",
            "Natural gas",
            "Renewables and waste",
            "Electricity",
            "Heat",
        ]

        values = []
        for key in expected_keys:
            if key in energy_mix:
                values.append(energy_mix[key])
            else:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Missing energy mix key: {key}, using 0.0")
                values.append(0.0)

        return torch.tensor([values], dtype=torch.float)

    def _get_energy_mix_by_country(self, country_name: str) -> torch.Tensor:
        """
        Get energy mix tensor by country name using the mapping from JSON file.

        Parameters
        ----------
        country_name : str
            Name of the country.

        Returns
        -------
        torch.Tensor
            Tensor representation of energy mix for the country.
        """
        energy_mapping = self._load_energy_mapping()
        country_lower = country_name.lower()

        if country_lower in energy_mapping:
            energy_values = energy_mapping[country_lower]
            return torch.tensor([energy_values], dtype=torch.float)
        else:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unknown country for energy mix: {country_name}, using default values"
            )
            # Return default energy mix (all zeros)
            return torch.tensor([[0.0] * 7], dtype=torch.float)

    @classmethod
    def create_single_molecule_data_static(
        cls,
        smiles: str,
        country_name: str = None,
        energy_mix: dict = None,
        dataset_type: str = "GNN_C",
    ) -> Data:
        """
        Create a single molecule data object from SMILES string.

        This method can be called without instantiating the class.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.
        country_name : str, optional
            Country name for GNN_C models or for energy mix lookup in GNN_E models.
        energy_mix : dict, optional
            Energy mix dictionary for GNN_E models. If not provided and country_name
            is given, will look up energy mix by country.
        dataset_type : str
            Type of dataset ("QSPR", "GNN_M", "GNN_C", "GNN_E").

        Returns
        -------
        Data
            PyTorch Geometric Data object for the molecule.
        """
        # Create a temporary instance to access feature extraction methods
        temp_instance = cls.__new__(cls)
        temp_instance.types = cls.types
        temp_instance.bonds = cls.bonds

        return temp_instance.create_single_molecule_data(
            smiles, country_name, energy_mix, dataset_type
        )


class QSPR_dataset(BaseMolecularDataset):
    """
    PyTorch Geometric dataset for QSPR (Quantitative Structure-Property Relationship) data.

    This dataset loads molecular descriptor data from Excel files and converts it into
    PyTorch Geometric format for machine learning applications. The dataset contains
    molecular features and Global Warming Impact (GWI) target values.

    The dataset processes 56 molecular descriptors including:
    - Basic molecular properties (MW, AMW, RBN, etc.)
    - Topological descriptors (D/Dtr, MAXDP, Yindex, etc.)
    - Physicochemical descriptors (P_VSA series, GATS, etc.)
    - 2D CATS descriptors
    - Fragment-based descriptors
    - MLOGP2 (octanol-water partition coefficient)

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    dataexcel : pandas.DataFrame
        DataFrame containing the molecular data with descriptors and GWI values.
    transform : callable, optional
        A function/transform that takes in a torch_geometric.data.Data object
        and returns a transformed version. The data object will be transformed
        before every access. Default is None.
    pre_transform : callable, optional
        A function/transform that takes in a torch_geometric.data.Data object
        and returns a transformed version. The data object will be transformed
        before being saved to disk. Default is None.
    pre_filter : callable, optional
        A function that takes in a torch_geometric.data.Data object and returns
        a boolean value, indicating whether the data object should be included
        in the final dataset. Default is None.

    Attributes
    ----------
    dataexcel : pandas.DataFrame
        The input molecular data DataFrame.
    data : torch_geometric.data.Data
        The processed dataset.
    slices : dict
        Dictionary containing slicing information for the dataset.
    """

    def __init__(
        self, root, dataexcel, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, dataexcel, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return the names of raw data files expected in the raw directory."""
        return ["2023_09_07_QSPR_mol_only_cc.xlsx"]

    @property
    def processed_file_names(self):
        """Return the names of processed data files to be saved in the processed directory."""
        return ["QSPR_Data.pt"]

    def process(self):
        """
        Process raw molecular data into PyTorch Geometric format.

        Read molecular descriptor data from the dataexcel DataFrame and convert it into
        torch_geometric.data.Data objects. Each molecule becomes a single data point
        with 56 molecular descriptors as features and GWI as the target value.

        The processed data is saved as a .pt file in the processed directory.
        To reprocess data, delete files in the processed folder.
        """
        data_list = []
        # Generating hashmaps
        X = torch.tensor(
            self.dataexcel[
                [
                    "MW",
                    "AMW",
                    "nBM",
                    "RBN",
                    "nF",
                    "N%",
                    "O%",
                    "D/Dtr05",
                    "D/Dtr10",
                    "MAXDP",
                    "Psi_i_A",
                    "Yindex",
                    "CIC4",
                    "CIC5",
                    "VR1_D/Dt",
                    "SpDiam_B(m)",
                    "ATSC2m",
                    "ATSC1p",
                    "GATS6m",
                    "GATS7s",
                    "P_VSA_LogP_1",
                    "P_VSA_LogP_2",
                    "P_VSA_LogP_8",
                    "P_VSA_MR_3",
                    "P_VSA_MR_5",
                    "P_VSA_MR_7",
                    "P_VSA_s_1",
                    "P_VSA_s_3",
                    "P_VSA_ppp_D",
                    "SpDiam_EA(dm)",
                    "SM14_AEA(dm)",
                    "SM15_AEA(dm)",
                    "SM02_AEA(ri)",
                    "SM04_AEA(ri)",
                    "SM06_AEA(ri)",
                    "SM10_AEA(ri)",
                    "nCp",
                    "nCs",
                    "H-046",
                    "H-047",
                    "H-051",
                    "H-052",
                    "SssO",
                    "CATS2D_02_DL",
                    "CATS2D_02_AA",
                    "CATS2D_02_AL",
                    "CATS2D_03_AL",
                    "CATS2D_05_AL",
                    "CATS2D_04_LL",
                    "T(N..Cl)",
                    "T(O..S)",
                    "T(O..Cl)",
                    "T(F..Cl)",
                    "F03[C-O]",
                    "F03[C-Cl]",
                    "MLOGP2",
                ]
            ].values,
            dtype=torch.float,
        )
        mol_id = torch.tensor(self.dataexcel["mol_id"].values, dtype=torch.float)
        Y = torch.tensor(self.dataexcel["GWI"].values, dtype=torch.float)

        for i in tqdm.trange(len(self.dataexcel)):
            x = X[i].unsqueeze(0)
            y = Y[i].unsqueeze(0)
            mol_num = mol_id[i].unsqueeze(0)

            data = Data(x=x, mol_id=mol_num, y=y)

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class GNN_M_dataset(BaseMolecularDataset):
    """
    PyTorch Geometric dataset for Graph Neural Network molecular data.

    This dataset converts SMILES strings into molecular graphs with node and edge features
    for Graph Neural Network applications. Each molecule is represented as a graph where
    atoms are nodes and bonds are edges, with rich featurization for both.
    """

    def __init__(
        self, root, dataexcel, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, dataexcel, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return the names of raw data files expected in the raw directory."""
        return ["2023_09_07_QSPR_mol_only_cc.xlsx"]

    @property
    def processed_file_names(self):
        """Return the names of processed data files to be saved in the processed directory."""
        return ["GNN_M_Data.pt"]

    def process(self):
        """
        Convert SMILES strings to molecular graphs with node and edge features.
        Uses base class utilities for common molecular processing operations.
        """
        mol_id = torch.tensor(self.dataexcel["mol_id"].values, dtype=torch.long)
        Y = torch.tensor(self.dataexcel["GWI"].values, dtype=torch.float)

        for i, smiles in enumerate(self.dataexcel["smiles"].values):
            mol = Chem.rdmolfiles.MolFromSmiles(smiles)
            mol = Chem.rdmolops.AddHs(mol)  # explicit trivial Hs (excluded)
            N = mol.GetNumAtoms()

            if N <= 1:
                x, edge_index, edge_attr = self._handle_single_atom_molecule()
            else:
                x = self._extract_atom_features(mol)
                edge_index, edge_attr = self._extract_bond_features(mol, N)

            y = Y[i].unsqueeze(0)
            mol_num = mol_id[i].unsqueeze(0)

            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, mol_id=mol_num, y=y
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            self.data_list.append(data)

        torch.save(self.collate(self.data_list), self.processed_paths[0])


class GNN_C_dataset(BaseMolecularDataset):
    """
    PyTorch Geometric dataset for Graph Neural Networks with country-specific environmental impact data.

    This dataset extends molecular graph representation by incorporating country information
    and multiple environmental impact targets from the Environmental Footprint (EF) v3.0
    methodology.
    """

    def __init__(
        self, root, dataexcel, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, dataexcel, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return the names of raw data files expected in the raw directory."""
        return ["2023_09_14_finaldataset_country_combine.xlsx"]

    @property
    def processed_file_names(self):
        """Return the names of processed data files to be saved in the processed directory."""
        return ["GNN_C_Data.pt"]

    def process(self):
        """Convert SMILES strings to molecular graphs with country context and multi-target environmental impacts."""
        data_list = []
        mol_id = torch.tensor(self.dataexcel["mol_id"].values, dtype=torch.float)

        # Environmental impact targets (15 categories)
        ef_columns = [
            "EF v3.0 - acidification - accumulated exceedance (ae) [mol H+-Eq]",
            "EF v3.0 - climate change: global warming potential (GWP100) [kg CO2-Eq]",
            "EF v3.0 - ecotoxicity: comparative toxic unit for ecosystems (CTUe)  [CTUe]",
            "EF v3.0 - energy resources: non-renewable - abiotic depletion potential (ADP): fossil fuels [MJ, net c.]",
            "EF v3.0 - eutrophication, freshwater - fraction of nutrients reaching freshwater end compartment (P) [kg PO4-Eq]",
            "EF v3.0 - eutrophication, marine - fraction of nutrients reaching marine end compartment (N) [kg N-Eq]",
            "EF v3.0 - eutrophication, terrestrial - accumulated exceedance (AE)  [mol N-Eq]",
            "EF v3.0 - human toxicity:  comparative toxic unit for human (CTUh)  [CTUh]",
            "EF v3.0 - ionising radiation: human health - human exposure efficiency relative to u235 [kBq U235-.]",
            "EF v3.0 - land use - soil quality index [dimension.]",
            "EF v3.0 - material resources: metals/minerals - abiotic depletion potential (ADP): elements (ultimate reserves) [kg Sb-Eq]",
            "EF v3.0 - ozone depletion - ozone depletion potential (ODP)  [kg CFC-11.]",
            "EF v3.0 - particulate matter formation - impact on human health [disease i.]",
            "EF v3.0 - photochemical ozone formation: human health - tropospheric ozone concentration increase [kg NMVOC-.]",
            "EF v3.0 - water use - user deprivation potential (deprivation-weighted water consumption) [m3 world .]",
        ]
        Y = torch.tensor(self.dataexcel[ef_columns].values, dtype=torch.float)

        country_id = torch.tensor(
            self.dataexcel["country_id"].values, dtype=torch.float
        )
        onehot_country_id = F.one_hot(country_id.long()).to(torch.float)

        for i, smiles in enumerate(self.dataexcel["smiles"].values):
            mol = Chem.rdmolfiles.MolFromSmiles(smiles)
            mol = Chem.rdmolops.AddHs(mol)
            N = mol.GetNumAtoms()

            if N <= 1:
                x, edge_index, edge_attr = self._handle_single_atom_molecule()
            else:
                x = self._extract_atom_features(mol)
                edge_index, edge_attr = self._extract_bond_features(mol, N)

            y = Y[i].unsqueeze(0)
            mol_num = mol_id[i].unsqueeze(0)
            country_num = onehot_country_id[i].unsqueeze(0)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mol_id=mol_num,
                country_id=country_num,
                y=y,
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class GNN_E_dataset(BaseMolecularDataset):
    """
    PyTorch Geometric dataset for Graph Neural Networks with energy mix-specific environmental impact data.

    This dataset extends molecular graph representation by incorporating energy mix information
    and multiple environmental impact targets from the Environmental Footprint (EF) v3.0
    methodology.
    """

    def __init__(
        self, root, dataexcel, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, dataexcel, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return the names of raw data files expected in the raw directory."""
        return ["2023_09_18_finaldataset_energymix_combine.xlsx"]

    @property
    def processed_file_names(self):
        """Return the names of processed data files to be saved in the processed directory."""
        return ["GNN_E_Data.pt"]

    def process(self):
        """Convert SMILES strings to molecular graphs with energy mix context and multi-target environmental impacts."""
        data_list = []
        mol_id = torch.tensor(self.dataexcel["mol_id"].values, dtype=torch.float)

        # Environmental impact targets (15 categories)
        ef_columns = [
            "EF v3.0 - acidification - accumulated exceedance (ae) [mol H+-Eq]",
            "EF v3.0 - climate change: global warming potential (GWP100) [kg CO2-Eq]",
            "EF v3.0 - ecotoxicity: comparative toxic unit for ecosystems (CTUe)  [CTUe]",
            "EF v3.0 - energy resources: non-renewable - abiotic depletion potential (ADP): fossil fuels [MJ, net c.]",
            "EF v3.0 - eutrophication, freshwater - fraction of nutrients reaching freshwater end compartment (P) [kg PO4-Eq]",
            "EF v3.0 - eutrophication, marine - fraction of nutrients reaching marine end compartment (N) [kg N-Eq]",
            "EF v3.0 - eutrophication, terrestrial - accumulated exceedance (AE)  [mol N-Eq]",
            "EF v3.0 - human toxicity:  comparative toxic unit for human (CTUh)  [CTUh]",
            "EF v3.0 - ionising radiation: human health - human exposure efficiency relative to u235 [kBq U235-.]",
            "EF v3.0 - land use - soil quality index [dimension.]",
            "EF v3.0 - material resources: metals/minerals - abiotic depletion potential (ADP): elements (ultimate reserves) [kg Sb-Eq]",
            "EF v3.0 - ozone depletion - ozone depletion potential (ODP)  [kg CFC-11.]",
            "EF v3.0 - particulate matter formation - impact on human health [disease i.]",
            "EF v3.0 - photochemical ozone formation: human health - tropospheric ozone concentration increase [kg NMVOC-.]",
            "EF v3.0 - water use - user deprivation potential (deprivation-weighted water consumption) [m3 world .]",
        ]
        Y = torch.tensor(self.dataexcel[ef_columns].values, dtype=torch.float)

        # Energy mix categories
        energy_columns = [
            "Coal, peat and oil shale",
            "Crude, NGL and feedstocks",
            "Oil products",
            "Natural gas",
            "Renewables and waste",
            "Electricity",
            "Heat",
        ]
        energy_id = torch.tensor(
            self.dataexcel[energy_columns].values, dtype=torch.float
        )

        for i, smiles in enumerate(self.dataexcel["smiles"].values):
            mol = Chem.rdmolfiles.MolFromSmiles(smiles)
            mol = Chem.rdmolops.AddHs(mol)
            N = mol.GetNumAtoms()

            if N <= 1:
                x, edge_index, edge_attr = self._handle_single_atom_molecule()
            else:
                x = self._extract_atom_features(mol)
                edge_index, edge_attr = self._extract_bond_features(mol, N)

            y = Y[i].unsqueeze(0)
            mol_num = mol_id[i].unsqueeze(0)
            energy_num = energy_id[i].unsqueeze(0)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mol_id=mol_num,
                energy_id=energy_num,
                y=y,
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
