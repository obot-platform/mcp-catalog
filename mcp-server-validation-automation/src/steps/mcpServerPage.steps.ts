import { When, Then } from '@wdio/cucumber-framework';
import { expect } from '@wdio/globals';

const selectors = {
    searchInput: '//input[@placeholder="Search servers..."]',

    serverRow: (name: string) =>
        `//p[normalize-space(text())="${name}"]/ancestor::tr`,

    actionMenu: (name: string) =>
        `//p[normalize-space(text())="${name}"]/ancestor::td/following-sibling::td[last()]//button`,

    menuActionBtn: (actionText: string) =>
        `//button[text()=" ${actionText}"]`,

    saveConfigBtn: '[data-testid="save-config"]',
    renameInput: '[data-testid="rename-input"]',
    renameConfirmBtn: '[data-testid="rename-confirm"]',
    successToast: '[data-testid="toast-success"]',
    backBtn: '[data-testid="back-to-server-list"]',
    disconnectedStatus: '[data-testid="server-status-disconnected"]',
    serverNameInDetails: '[data-testid="server-name"]'
};


When(
    /^User searches for MCP server "([^"]*)"$/,
    async (serverName: string) => {
        const input = await $(selectors.searchInput);
        await input.waitForDisplayed();
        await input.setValue(serverName);
    }
);

When(
    /^User performs "([^"]*)" action on MCP server "([^"]*)"$/,
    async (action: string, serverName: string) => {
        await $(selectors.actionMenu(serverName)).click();
        await $(selectors.menuActionBtn(action)).click();
    }
);


When(
    /^User updates MCP server config and saves$/,
    async () => {
        await $(selectors.saveConfigBtn).click();
    }
);

Then(
    /^MCP server config should be updated successfully$/,
    async () => {
        await expect($(selectors.successToast)).toBeDisplayed();
    }
);

When(
    /^User updates MCP server name to "([^"]*)"$/,
    async (newName: string) => {
        const input = await $(selectors.renameInput);
        await input.waitForDisplayed();
        await input.setValue(newName);
        await $(selectors.renameConfirmBtn).click();
    }
);

Then(
    /^MCP server name should be updated successfully$/,
    async () => {
        await expect($(selectors.successToast)).toBeDisplayed();
    }
);

Then(
    /^MCP server details should be correct$/,
    async () => {
        await expect($(selectors.serverNameInDetails)).toBeDisplayed();
    }
);

When(
    /^User navigates back to MCP server list$/,
    async () => {
        await $(selectors.backBtn).click();
    }
);

Then(
    /^MCP server should be disconnected successfully$/,
    async () => {
        await expect($(selectors.disconnectedStatus)).toBeDisplayed();
    }
);
