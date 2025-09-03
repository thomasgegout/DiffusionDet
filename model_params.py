import train_net
import json

if __name__ == "__main__":
    args = train_net.default_argument_parser().parse_args()
    cfg = train_net.setup(args)
    model = train_net.Trainer.build_model(cfg)
    json.dump( {n: str(type(m)) for n, m in model.named_modules()}, open("model_params.json", "w"), indent=2)
