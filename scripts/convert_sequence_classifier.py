import argparse
import sys

from transformers import BartForSequenceClassification

from transformers_bart_finetune.models import TFBartForSequenceClassification

# fmt: off
parser = argparse.ArgumentParser(description='Convert "BartForSequenceClassification" and "TFBartForSequenceClassification"')
parser.add_argument("--pretrained-model", required=True, help="pretrained model path")
parser.add_argument("--output-path", required=True, help="output path")
parser.add_argument("--to", required=True, choices=["tf", "torch"], help='output checkpoint format. "tf" or "torch"')
# fmt: on


def main(args: argparse.Namespace):
    import transformers

    # Set non-existing TFBartForSequenceClassification class
    transformers.TFBartForSequenceClassification = TFBartForSequenceClassification

    if args.to == "torch":
        model = BartForSequenceClassification.from_pretrained(args.pretrained_model, from_tf=True)
    elif args.to == "tf":
        model = TFBartForSequenceClassification.from_pretrained(args.pretrained_model, from_pt=True)
    model.save_pretrained(args.output_path)
    print(f'[+] Save pretrained model to "{args.output_path}"')


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
