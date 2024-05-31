import argparse
from model.QTTModel import QTTModel
from model.CPModel import CPModel
from model.VMModel import TensorVM
from model.TuckerModel import TuckerModel
from utils import *


# Example usage:
# python src/get_sizes_of_models.py --model QTT --rank 200 --target cube256 --increase_rank_factor_CP 50 --increase_rank_factor_VM 2 --increase_rank_factor_Tucker 10


def main():
    parser = argparse.ArgumentParser(description="Tensor Compression Script")
    parser.add_argument("--model", type=str, required=True, help="Model type (QTT, CP, VM)")
    parser.add_argument("--rank", type=int, required=True, help="Rank of QTT")
    parser.add_argument("--target", type=str, required=True, help="Target data")
    parser.add_argument("--increase_rank_factor_CP", type=int, default=1, help="Factor to increase rank for CP")
    parser.add_argument("--increase_rank_factor_VM", type=int, default=1, help="Factor to increase rank for VM")
    parser.add_argument("--increase_rank_factor_Tucker", type=int, default=1, help="Factor to increase rank for Tucker")
    args = parser.parse_args()

    target = args.target
    payload_position = "grayscale"
    payload = 0
    dimensions = 3  # Default to 3D

    if dimensions == 2:
        target = get_target_image(target, payload_position, normalization="min_max")
    if dimensions == 3:
        target = get_target_3d_object(target, payload_position, normalization="min_max")
    reso = target.shape[0]
    dimensions = len(target.shape)

    if args.model == "QTT":
        model = QTTModel(target, init_reso=reso, max_rank=args.rank, loss_fn_str="L2", activation="None",
                         regularization_type="TV", dimensions=dimensions, payload_position=payload_position, payload=payload)
    else:
        raise ValueError("Invalid model type. Use 'QTT'.")

    # Get compression variables for QTT
    qtt_size_uncompressed = model.sz_uncompressed_gb
    qtt_size_compressed = model.sz_compressed_gb
    qtt_compression_factor = model.compression_factor

    print("********************************")
    print(f"Model: {args.model}")
    print(f"Max Rank (QTT): {args.rank}")
    print(f"Size Uncompressed (QTT): {qtt_size_uncompressed} GB")
    print(f"Size Compressed (QTT): {qtt_size_compressed} GB")
    print(f"Compression Factor (QTT): {qtt_compression_factor}")
    print("********************************")

    current_rank_CP = 0
    size_compressed_CP = 0
    current_rank_VM = 0
    size_compressed_VM = 0
    current_rank_Tucker = 0
    size_compressed_Tucker = 0
    
    while True:
        # Increase rank for each model if necessary
        if size_compressed_CP < qtt_size_compressed:
            current_rank_CP += args.increase_rank_factor_CP
        if  size_compressed_VM < qtt_size_compressed:
            current_rank_VM += args.increase_rank_factor_VM
        if size_compressed_Tucker < qtt_size_compressed:
            current_rank_Tucker += args.increase_rank_factor_Tucker
        
        model = CPModel(target, init_reso=reso, max_rank=current_rank_CP, loss_fn_str="L2", activation="None",
                        regularization_type="TV", payload_position=payload_position, dimensions=dimensions, payload=payload)
        size_compressed_CP = model.sz_compressed_gb

        if dimensions > 2: 
            model = TensorVM(target, init_reso=reso, max_rank=current_rank_VM, loss_fn_str="L2", activation="None",
                            regularization_type="TV", payload_position=payload_position, dimensions=dimensions, payload=payload)
            size_compressed_VM = model.sz_compressed_gb
        
        model = TuckerModel(target, init_reso=reso, max_rank=current_rank_Tucker, loss_fn_str="L2", activation="None",
                            regularization_type="TV", payload_position=payload_position, dimensions=dimensions, payload=payload)
        size_compressed_Tucker = model.sz_compressed_gb

        print("********************************")
        print(f"CP rank: {current_rank_CP}")
        print(f"Size Compressed (CP): {size_compressed_CP} GB")
        print("********************************")
        print(f"VM rank: {current_rank_VM}")
        print(f"Size Compressed (VM): {size_compressed_VM} GB")
        print("********************************")
        print(f"Tucker rank: {current_rank_Tucker}")
        print(f"Size Compressed (Tucker): {size_compressed_Tucker} GB")
        print("********************************")
        
        if size_compressed_CP > qtt_size_compressed and size_compressed_VM > qtt_size_compressed and size_compressed_Tucker > qtt_size_compressed:
            break   

if __name__ == "__main__":
    main()
