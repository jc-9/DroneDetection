# Function to check if a substring is a palindrome
def is_palindrome(s):
    return s == s[::-1]


# Function to find all palindromes in a string
def find_all_palindromes(input_string):
    palindromes = set()  # Using a set to avoid duplicates

    # Check all possible substrings
    for i in range(len(input_string)):
        for j in range(i + 1, len(input_string) + 1):
            substring = input_string[i:j]
            if is_palindrome(substring):  # Check if it's a palindrome
                palindromes.add(substring)

    return list(palindromes)


# Example usage
input_string = "7100303001"
palindromes = find_all_palindromes(input_string)
print(palindromes)
