import yaml
import pandas as pd
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel


class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        checkpoint_dir = config["model_checkpoint_path"]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=config.get("max_seq_length", 256),
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(self.model)

        # Danh sách 77 ý định được làm sạch và định dạng dưới dạng danh sách
        self.categories = [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            "atm_support",
            "automatic_top_up",
            "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit",
            "beneficiary_not_allowed",
            "cancel_transfer",
            "card_about_to_expire",
            "card_acceptance",
            "card_arrival",
            "card_delivery_estimate",
            "card_linking",
            "card_not_working",
            "card_payment_fee_charged",
            "card_payment_not_recognised",
            "card_payment_wrong_exchange_rate",
            "card_swallowed",
            "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised",
            "change_pin",
            "compromised_card",
            "contactless_not_working",
            "country_support",
            "declined_card_payment",
            "declined_cash_withdrawal",
            "declined_transfer",
            "direct_debit_payment_not_recognised",
            "disposable_card_limits",
            "edit_personal_details",
            "exchange_charge",
            "exchange_rate",
            "exchange_via_app",
            "extra_charge_on_statement",
            "failed_transfer",
            "fiat_currency_support",
            "get_disposable_virtual_card",
            "get_physical_card",
            "getting_spare_card",
            "getting_virtual_card",
            "lost_or_stolen_card",
            "lost_or_stolen_phone",
            "order_physical_card",
            "passcode_forgotten",
            "pending_card_payment",
            "pending_cash_withdrawal",
            "pending_top_up",
            "pending_transfer",
            "pin_blocked",
            "receiving_money",
            "Refund_not_showing_up",
            "request_refund",
            "reverted_card_payment",
            "supported_cards_and_currencies",
            "terminate_account",
            "top_up_by_bank_transfer_charge",
            "top_up_by_card_charge",
            "top_up_by_cash_or_cheque",
            "top_up_failed",
            "top_up_limits",
            "top_up_reverted",
            "topping_up_by_card",
            "transaction_charged_twice",
            "transfer_fee_charged",
            "transfer_into_account",
            "transfer_not_received_by_recipient",
            "transfer_timing",
            "unable_to_verify_identity",
            "verify_my_identity",
            "verify_source_of_funds",
            "verify_top_up",
            "virtual_card_not_working",
            "visa_or_mastercard",
            "why_verify_identity",
            "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal",
        ]

        # Chuyển đổi danh sách thành chuỗi có các mục được phân tách bằng dấu phẩy
        self.categories_str = ", ".join(self.categories)

        # Cập nhật prompt template với danh sách categories
        self.prompt_template = """Below is a customer message to a bank. Classify the intent of the message into the correct category from the list below.

### Categories:
{categories}

### Message:
{message}

### Intent:
"""

    def __call__(self, message):
        formatted_prompt = self.prompt_template.format(
            categories=self.categories_str, message=message
        )
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

        # Giữ nguyên giới hạn sinh token và ép dừng
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=15,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ]

        predicted_text = decoded_output.split("### Intent:\n")[-1].strip()

        # Giữ nguyên xử lý hậu kỳ chống over-generation
        predicted_label = predicted_text.split("\n")[0].strip()

        return predicted_label


if __name__ == "__main__":
    classifier = IntentClassification(model_path="configs/inference.yaml")

    with open("configs/inference.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    test_data_path = config["test_path"]
    print(f"Loading test data from {test_data_path}...")

    try:
        df_test = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(f"File not found: {test_data_path}")
        exit()

    y_true = df_test["intent"].tolist()
    messages = df_test["text"].tolist()
    y_pred = []

    total_samples = len(messages)
    print(f"Evaluating on {total_samples} samples...")

    for idx, msg in enumerate(messages):
        pred = classifier(msg)
        y_pred.append(pred)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
            print(f"Processed {idx + 1}/{total_samples} samples...")

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
