# ML4
# FER2013 — Facial Expression Recognition

## Competition Overview
პროექტი დაფუძნებულია კაგლის ქომფეთიშენზე "Challenges in Representation Learning: Facial Expression Recognition Challenge." მიზანია სახის გამომეტყველების კლასიფიცირება 7 ემოციური კატეგორიიდან ერთ-ერთში: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. სურათები არის 48×48 პიქსელის გრეისქეილ ფორმატში. ვიყენებთ **accuracy**-ს შეფასების მეტრიკად.

WandB პროექტი: https://wandb.ai/ggzob23-free-university-of-tbilisi-/fer2013-facial-expression

---

## Approach
მაღალი პერფორმანსისთვის და ნეირონული ქსელების ქცევის დემონსტრაციისთვის, გავტესტეთ რამდენიმე არქიტექტურა — დავიწყეთ უმარტივესი მოდელიდან და იტერაციულად გავზარდეთ სირთულე. გადაიდგა შემდეგი ნაბიჯები:

- მონაცემების ჩატვირთვა და ექსპლორაცია
- Data Augmentation და პრეპროცესინგი
- Forward და Backward შემოწმებები
- მოდელის ტრეინინგი არქიტექტურის იტერაციული გაუმჯობესებით
- ჰიპერპარამეტრების ოპტიმიზაცია თითოეული არქიტექტურისთვის
- Overfitting/Underfitting ანალიზი
- საბოლოო მოდელის შერჩევა

---

## Repository Structure
├── ML4.ipynb       # Main notebook with all experiments

└── README.md       # This file


---

## Dataset Analysis
**ტრეინინგი:** 28,709 სურათი | **ტესტი:** 7,178 სურათი

დატასეტი მნიშვნელოვნად დაუბალანსებელია:

| Emotion | Train | Test |
|---------|-------|------|
| Happy | 7,215 | 1,774 |
| Neutral | 4,965 | 1,233 |
| Sad | 4,830 | 1,247 |
| Fear | 4,097 | 1,024 |
| Angry | 3,995 | 958 |
| Surprise | 3,171 | 831 |
| Disgust | 436 | 111 |

ეს დისბალანსი პირდაპირ აისახება შედეგებზე — Happy კლასი ყველა მოდელში საუკეთესო F1-ს იღებს, Disgust კი ყველაზე ცუდს, ძალიან მცირე სატრეინინგო მაგალითების გამო.

---

## Forward & Backward Checks
ნებისმიერი მოდელის ტრეინინგამდე ჩავატარეთ ორი კრიტიკული შემოწმება, როგორც განხილული იყო ლექციებზე:

**Forward Check:**  4 სურათის ბატჩი გავატარეთ თითოეულ მოდელში და დავადასტურეთ, რომ გამოსავლის ზომაა `[4, 7]` — ეს ადასტურებს სწორ არქიტექტურას 7-კლასიანი კლასიფიკაციისთვის.

**Backward Check:** ერთი forward pass-ისა და loss-ის გამოთვლის შემდეგ, დავადასტურეთ, რომ ყველა trainable პარამეტრმა მიიღო არანულოვანი გრადიენტი. ეს ადასტურებს:
- გრადიენტები მიედინება ყველა layer-ში
- Dead layer-ები არ არსებობს ქსელში
- ResNetMini-ს skip connections სწორად გადასცემს გრადიენტებს residual path-ის გავლით

ყველა არქიტექტურამ გაიარა ორივე შემოწმება ტრეინინგამდე.

---

## Data Augmentation
სატრეინინგო მონაცემები გავამდიდრეთ განზოგადების გასაუმჯობესებლად:
- `RandomHorizontalFlip` — სახეები სიმეტრიულია, გადაბრუნება ვალიდურია
- `RandomRotation(10°)` — მცირე როტაცია ბუნებრივი თავის დახრის სიმულაციაა
- `ColorJitter(brightness=0.2, contrast=0.2)` — სხვადასხვა განათების სიმულაცია
- `Normalize(mean=0.5, std=0.5)` — პიქსელების ნორმალიზაცია სტაბილური ტრეინინგისთვის

ვალიდაციისა და ტესტის სეტებზე გამოიყენება მხოლოდ ToTensor + Normalize (augmentation გარეშე).
---

## Architectures & Experiments

### არქიტექტურა 1 — TinyMLP (განზრახ Underfit)
მარტივი MLP ერთი hidden layer-ით 128 ნეირონით. შეყვანის სურათები ვაბრტყელებთ 48×48-დან 2304-განზომილებიან ვექტორად.

**არჩევის მიზეზი:** საბაზისო მოდელი underfitting-ის დემონსტრაციისთვის. Fully connected ქსელი фიგნორებს სივრცულ სტრუქტურას — თვლის (0,0) და (47,47) პიქსელებს თანაბრად დაკავშირებულად, რაც სახის სურათებისთვის მცდარია.


| Metric | Value |
|--------|-------|
| Train Acc | 45.29% |
| Val Acc | 40.52% |
| Test Acc | 41.70% |
| Overfit Gap | ~4.7% |

**ანალიზი:** Train და val accuracy ორივე დაბალია, მცირე gap-ით — კლასიკური underfitting. მოდელს არ აქვს სიმძლავრე სივრცული ფიჩერების სასწავლად. გადაწყვეტილება: CNN-ზე გადასვლა.

---

### Architecture 2 — SmallCNN
Two convolutional layers (32→64 filters), ReLU activations, MaxPool2d, followed by a fully connected head with Dropout(0.3) and L2 weight decay.

**არჩევის მიზეზი:** პირველი ნაბიჯი CNN-ებში. კონვოლუციური layer-ები სწავლობენ ლოკალურ ფილტრებს (კიდეები, მრუდები, ტექსტურები), რაც სახის ემოციების ამოცნობისთვის პირდაპირ რელევანტურია.

| Metric | Value |
|--------|-------|
| Train Acc | ~67% |
| Val Acc | ~54% |
| Test Acc | 55.98% |
| Overfit Gap | ~13% |

**ანალიზი:** +14%-იანი ნახტომი TinyMLP-თან შედარებით ადასტურებს, რომ სივრცული ფიჩერების სწავლება კრიტიკულია ამ ამოცანისთვის. გამოჩნდა ზომიერი overfitting gap. გადაწყვეტილება: მეტი layer-ების დამატება სიღრმის გასაზრდელად.

---

### არქიტექტურა 3 — DeepCNN (Overfitting-ის დემონსტრაცია)
ოთხი კონვოლუციური layer (32→64→128→256 ფილტრი), BatchNorm2d, MaxPool2d. FC head Dropout(0.5) და Dropout(0.3)-ით. **L2 რეგულარიზაციის გარეშე, scheduler-ის გარეშე** — განზრახ overfitting-ის ჩვენებისთვის.

**არჩევის მიზეზი:** ვაჩვენოთ რა ხდება სიმძლავრის გაზრდისას საკმარისი რეგულარიზაციის გარეშე. იგივე არქიტექტურა გამეორდება რეგულარიზაციით შემდეგ ექსპერიმენტში იზოლირებული შედარებისთვის.

| Metric | Value |
|--------|-------|
| Train Acc | 72.86% |
| Val Acc | 59.52% |
| Test Acc | 62.83% |
| Overfit Gap | ~13.3% |

**ანალიზი:** მკაფიო overfitting — train accuracy val-ს 13%-ით სჯობს. მოდელი სატრეინინგო მონაცემებს ზეპირად იმახსოვრებს. გამოსწორება: L2 რეგულარიზაცია და LR scheduler.

---

### არქიტექტურა 3b — DeepCNN რეგულარიზებული
იდენტური DeepCNN არქიტექტურა weight_decay=1e-3 და StepLR scheduler-ით (step_size=10, gamma=0.5). LR შემცირდა 0.0005-მდე.

**არჩევის მიზეზი:** A/B შედარება arch3-თან რეგულარიზაციის ეფექტის იზოლირებისთვის. ყველა სხვა ჰიპერპარამეტრი იდენტურია.

| Metric | Value |
|--------|-------|
| Train Acc | 76.50% |
| Val Acc | 60.95% |
| Test Acc | **64.03%** |
| Overfit Gap | ~15.5% |

**ანალიზი:** საუკეთესო მოდელი საერთო ჯამში. Test accuracy გაიზარდა 1.2%-ით რეგულარიზაციის გარეშე ვარიანტთან შედარებით. LR schedule მკაფიოდ ჩანს WandB ლოგებში — LR მცირდება 0.0005-დან 3e-05-მდე 40 ეპოქის განმავლობაში.

---

### არქიტექტურა 4 — ResNetMini SGD vs Adam
Custom mini-ResNet stem layer-ით, 3 residual block-ით (skip connection x + F(x)), AdaptiveAvgPool2d-ით და linear classification head-ით.

**არჩევის მიზეზი:** Skip connections წყვეტს vanishing gradient პრობლემას ღრმა ქსელებში. ეს დადასტურდა backward check-ით — ყველა layer-მა, residual block-ების შიგნით ჩათვლით, მიიღო ნულოვანი გრადიენტი. ვტესტავთ ორ optimizer-ს იმავე არქიტექტურაზე კონვერგენციის შედარებისთვის.

| Metric | SGD | Adam |
|--------|-----|------|
| Train Acc | 74.76% | 72.58% |
| Val Acc | 61.47% | 60.65% |
| Test Acc | 62.91% | 62.76% |
| Overfit Gap | ~13.3% | ~12% |

**ანალიზი:** ორივე optimizer-მა მიაღწია თითქმის იდენტურ საბოლოო accuracy-ს (~62.8%). Adam-მა უფრო სწრაფად დაიწყო კონვერგენცია (ჩანს WandB loss curves-ში ადრეულ ეპოქებში), მაგრამ SGD-მა ოდნავ გაასწრო საბოლოო test accuracy-ში. ResNetMini-მა ვერ გაასწრო რეგულარიზებულ DeepCNN-ს, სავარაუდოდ იმიტომ, რომ 48×48 სურათებისთვის 4 conv layer საკმარისია.

---

## Results Summary

| # | Experiment | Test Acc | Overfit Gap | Key Finding |
|---|---|---|---|---|
| 1 | TinyMLP | 41.70% | ~4.7% | Underfitting — სივრცული ფიჩერები არ ისწავლება |
| 2 | SmallCNN | 55.98% | ~13% | CNN-ები სივრცულ ფიჩერებს სწავლობენ |
| 3 | DeepCNN (overfit) | 62.83% | ~13.3% | სიღრმე ეხმარება, მაგრამ overfitting-ი ჩნდება |
| 4 | DeepCNN (regularized) | **64.03%** | ~15.5% | L2 + scheduler = საუკეთესო განზოგადება |
| 5 | ResNetMini SGD | 62.91% | ~13.3% | Skip connections გრადიენტს ეხმარება |
| 6 | ResNetMini Adam | 62.76% | ~12% | Adam სწრაფია, საბოლოო შედეგი იგივეა |

---

## Hyperparameter Tuning Strategy

**Learning Rate:** TinyMLP და SmallCNN — 0.001 (Adam-ის სტანდარტული default). DeepCNN regularized — 0.0005 (დაბალი scheduler-თან კომბინაციაში). ResNetMini SGD — 0.01 (SGD-ს უფრო მაღალი LR სჭირდება). ResNetMini Adam — 0.0003.

**Weight Decay (L2):** TinyMLP — 0 (baseline). SmallCNN — 1e-4 (მსუბუქი). DeepCNN overfit — 0 (განზრახ). DeepCNN regularized — 1e-3 (ძლიერი). ResNetMini — 1e-4.

**LR Scheduler:** StepLR(step=10, gamma=0.5) — SmallCNN, DeepCNN regularized, ორივე ResNetMini. TinyMLP და DeepCNN overfit-ზე არ გამოვიყენეთ ქცევის იზოლირებისთვის.

**Dropout:** Dropout(0.3) SmallCNN-ში. Dropout(0.5) + Dropout(0.3) DeepCNN და ResNetMini-ს head-ში.

---

## Key Findings

**Underfitting:** TinyMLP (41.70%) — train და val ორივე დაბალია, მცირე gap. მიზეზი: სივრცული ფიჩერების ექსტრაქცია არ ხდება. გამოსწორება: CNN-ზე გადასვლა.

**Overfitting:** DeepCNN რეგულარიზაციის გარეშე — train 72.86% vs val 59.52%, 13.3%-იანი gap. მიზეზი: მაღალი სიმძლავრის მოდელი L2 penalty-ს გარეშე. გამოსწორება: weight_decay=1e-3 + LR scheduler.

**სიღრმის ეფექტი:** მკაფიო პროგრესია TinyMLP→SmallCNN→DeepCNN: 41.70%→55.98%→62.83%. თითოეული დამატებული layer სწავლობს უფრო მაღალ დონის ფიჩერებს.

**Optimizer-ების შედარება:** Adam უფრო სწრაფად კონვერგირებს ადრეულ ეპოქებში, SGD კი momentum-ით ოდნავ უკეთეს საბოლოო შედეგს აღწევს. ამ დატასეტზე optimizer-ის არჩევანი ნაკლებად მნიშვნელოვანია ვიდრე არქიტექტურა და რეგულარიზაცია.

**კლასების დისბალანსი:** Disgust (111 ტესტის მაგალითი) F1≈0.20-0.46 ყველა მოდელში. Happy (1774 მაგალითი) F1≈0.60-0.84. მოდელი სწავლობს იმას, რასაც ყველაზე ხშირად ხედავს.

**Forward და Backward შემოწმებები:** ყველა არქიტექტურამ გაიარა. ResNetMini-ს skip connections-მა backward check-ზე დაადასტურა სწორი გრადიენტის გადაცემა residual path-ის გავლით.

---

## WandB Tracking
თითოეული ექსპერიმენტი დალოგილია ცალკე run-ად შემდეგი მონაცემებით:
- თითო ეპოქაზე: `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr`
- ბოლოს: `test_acc`, `test_loss`, `best_val_acc`, `confusion_matrix` სურათი

