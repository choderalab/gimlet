
def test_read_oeb():
    """Test load oemol from oeb file"""
    from openeye import oechem
    from gin.i_o.utils import file_to_oemols
    filename = 'data/benzoic_acid.oeb'
    oemol = file_to_oemols(filename)
    assert isinstance(oemol, oechem.OEMol)

def test_oemol_to_dict():
    """ Test write oemol to dict of partial charges and connectivity"""

    from gin.i_o.utils import file_to_oemols, oemol_to_dict
    filename = 'data/benzoic_acid.oeb'
    oemol = file_to_oemols(filename)

    oemol_dict = oemol_to_dict(oemol)
    assert oemol_dict['atomic_symbols'] == ['C','C','C','C','C','C','C','O','O','H','H','H','H', 'H','H']
    assert oemol_dict['partial_charges'] == [-0.09494999796152115, -0.1476300060749054, -0.1476300060749054,
                                             -0.06898000091314316, -0.06898000091314316, -0.1377900093793869,
                                              0.6511199474334717,  -0.5530099868774414, -0.6089699864387512,
                                              0.1360200047492981,   0.14047999680042267,  0.14047999680042267,
                                              0.156810000538826,    0.156810000538826, 0.4462200105190277]
    assert oemol_dict['connectivity'] == [[0, 1, 1.4098186492919922],
                                          [0, 2, 1.4120616912841797],
                                          [1, 3, 1.4274444580078125],
                                          [2, 4, 1.4262700080871582],
                                          [3, 5, 1.374497890472412],
                                          [4, 5, 1.377168893814087],
                                          [5, 6, 0.9570216536521912],
                                          [6, 7, 1.7774628400802612],
                                          [6, 8, 1.0434139966964722],
                                          [0, 9, 0.9487380981445312],
                                          [1, 10, 0.9474759101867676],
                                          [2, 11, 0.9475669860839844],
                                          [3, 12, 0.9424725770950317],
                                          [4, 13, 0.9428789019584656],
                                          [8, 14, 0.9101700186729431]]


    atoms = oemol_dict['atomic_symbols']
    atoms = tf.expand_dims(tf.convert_to_tensor(
            atoms,
            tf.string),
        1)
    atoms = tf.cast(
        tf.map_fn(
            lambda x: TRANSLATION[x.numpy()[0]],
            atoms,
            tf.int32),
        tf.int64)

    atoms = tf.reshape(
        atoms,
        [-1])

    n_atoms = tf.shape(atoms, tf.int64)[0]

    bonds = tf.convert_to_tensor(
        oemol_dict['connectivity'],
        dtype=tf.float32)

    adjacency_map = tf.zeros(
        (n_atoms, n_atoms),
        tf.float32)

    adjacency_map = tf.tensor_scatter_nd_update(
        adjacency_map,

        tf.cast(
            bonds[:, :2],
            tf.int64),

        bonds[:, 2])

    adjacency_map = gin.i_o.utils.conjugate_average(atoms, adjacency_map)

    charges = tf.convert_to_tensor(
        oemol_dict['partial_charges'],
        tf.float32)

test_oemol_to_dict()        
